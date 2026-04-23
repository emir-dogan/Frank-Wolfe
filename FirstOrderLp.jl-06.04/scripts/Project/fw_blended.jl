# ------------------------------------------------------------
# Blended Conditional Gradients (BCG)
#
# Maintains x as a convex combination of atoms (active set).
# Uses a "lazy" threshold Phi to avoid calling the LMO.
# Each iteration:
#   1. Inner Gap >= Phi: Simplex Step (Local FW step within S)
#   2. Inner Gap < Phi:  Global LMO call
#      a. FW Gap >= Phi: Standard FW step, add to S
#      b. FW Gap < Phi:  Halve Phi, skip step
# ------------------------------------------------------------

# Objective and gradient
function f(Q, c, x)
    return 0.5 * dot(x, Q * x) + dot(c, x)
end

function grad_f(Q, c, x)
    return Q * x + c
end

function blended_frank_wolfe(Q, c, A, b;time_limit=300.0, max_iter=100, tol=1e-6, verbose=true,
                              stepsize=:exact,
                              x0=nothing, atoms0=nothing, weights0=nothing)
    max_iter = isinf(max_iter) ? typemax(Int) : Int(max_iter)
    n = length(c)

    # BUG FIX 2: random direction avoids degenerate Phi=0 initialization
    if atoms0 !== nothing && weights0 !== nothing
        atoms   = copy.(atoms0)
        weights = copy(weights0)
        x       = sum(weights[i] .* atoms[i] for i in eachindex(atoms))
    elseif x0 !== nothing
        atoms   = [copy(x0)]
        weights = [1.0]
        x       = copy(x0)
    else
        init_s  = lmo_firstorderlp(randn(n), A, b; iter_limit=300000, verbose=0)
        atoms   = [copy(init_s)]
        weights = [1.0]
        x       = copy(init_s)
    end

    hist_obj = Float64[]
    hist_gap = Float64[]

    g_init = grad_f(Q, c, x)
    s_init = lmo_firstorderlp(g_init, A, b; iter_limit=300000, verbose=0)
    Phi    = dot(g_init, x - s_init)
    # BUG FIX 2: guard against zero/negative Phi from degenerate init
    Phi    = max(Phi, tol * 10)

    iter_no = 0
    start_time = time()
    for t in 1:max_iter
        if time() - start_time > time_limit
            println("Terminating: Time limit of $time_limit seconds exceeded.")
            break
        end
        iter_no += 1
        g = grad_f(Q, c, x)

        gv             = [dot(g, a) for a in atoms]
        best_inner_idx = argmin(gv)
        v_S            = atoms[best_inner_idx]
        g_S            = dot(g, x) - gv[best_inner_idx]

        push!(hist_obj, f(Q, c, x))
        push!(hist_gap, Phi)

        if Phi <= tol
            verbose && println("iter=$(lpad(t,4)), [Done], obj=$(round(hist_obj[end],digits=6)), phi=$(round(Phi,digits=6))")
            break
        end

        step_type = "None"  # always initialized now

        if g_S >= Phi
            # ------ Simplex Step ------
            step_type = "Simp"
            d       = v_S - x

            if norm(d) < 1e-14
                Phi /= 2.0  # stuck inside active set, shrink threshold
            else
                eta_max = 1.0
                if stepsize == :exact_bounded || stepsize == :exact
                    η = clamp(exact_stepsize_bounded(Q, c, x, d, eta_max), 0.0, eta_max)
                elseif stepsize == :diminishing
                    η = min(diminishing_stepsize(t), eta_max)
                else
                    η = clamp(armijo_stepsize(Q, c, x, d, g; eta_init=eta_max), 0.0, eta_max)
                end
                weights .*= (1 - η)
                weights[best_inner_idx] += η
            end

        else
            # ------ Global LMO call ------
            v_FW = lmo_firstorderlp(g, A, b; iter_limit=300000, verbose=0)
            g_FW = dot(g, x - v_FW)

            if g_FW >= Phi
                step_type = "FW  "
                d       = v_FW - x
                eta_max = 1.0

                if stepsize == :exact_bounded || stepsize == :exact
                    η = clamp(exact_stepsize_bounded(Q, c, x, d, eta_max), 0.0, eta_max)
                elseif stepsize == :diminishing
                    η = min(diminishing_stepsize(t), eta_max)
                else
                    η = clamp(armijo_stepsize(Q, c, x, d, g; eta_init=eta_max), 0.0, eta_max)
                end

                weights .*= (1 - η)
                idx = findfirst(a -> norm(a - v_FW) < 1e-10, atoms)
                if idx === nothing
                    push!(atoms, copy(v_FW))
                    push!(weights, η)
                else
                    weights[idx] += η
                end

                # BUG FIX 1: refresh Phi to the true measured gap.
                # Phi only decreases via halving; after an FW step we know the
                # real gap is g_FW, so using it keeps Phi as tight as possible
                # and prevents over-reliance on cheap-but-stagnating simplex steps.
                Phi = g_FW

            else
                step_type = "Drop"
                Phi /= 2.0
            end
        end

        verbose && t % 100 == 0 && println(
            "iter=$(lpad(t,4)), [$step_type], obj=$(round(hist_obj[end],digits=6)), " *
            "phi=$(round(Phi,digits=6)), g_S=$(round(g_S,digits=6))")

        keep    = weights .> 1e-12
        atoms   = atoms[keep]
        weights = weights[keep]
        weights ./= sum(weights)

        x = sum(weights[i] .* atoms[i] for i in eachindex(atoms))
    end

    return x, hist_obj, hist_gap, iter_no
end