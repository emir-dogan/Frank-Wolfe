# ------------------------------------------------------------
# Blended Pairwise Conditional Gradients (BPCG)
#
# Uses a "lazy" threshold Phi to avoid calling the LMO.
# Each iteration:
#   1. Pairwise Gap >= Phi: Pairwise Step (Strictly within active set)
#   2. Pairwise Gap < Phi:  Global LMO call
#      a. FW Gap >= Phi: Standard FW step, add to active set
#      b. FW Gap < Phi:  Halve Phi (Drop Step)
# ------------------------------------------------------------

# Objective and gradient
function f(Q, c, x)
    return 0.5 * dot(x, Q * x) + dot(c, x)
end

function grad_f(Q, c, x)
    return Q * x + c
end

function blended_pairwise_frank_wolfe(Q, c, A, b; max_iter=100, tol=1e-6, verbose=true,
                                      stepsize=:exact, time_limit=300.0, lmo_iter=3000) 
    max_iter = isinf(max_iter) ? typemax(Int) : Int(max_iter)
    n = length(c)

    # 1. Initialize strictly feasible using the LMO
    init_s  = lmo_firstorderlp(c, A, b; iter_limit=lmo_iter, verbose=0)
    atoms   = [copy(init_s)]
    weights = [1.0]
    x       = copy(init_s)

    hist_obj = Float64[]
    hist_gap = Float64[]

    # Initialize threshold Phi using the very first global FW gap
    g_init = grad_f(Q, c, x)
    s_init = lmo_firstorderlp(g_init, A, b; iter_limit=lmo_iter, verbose=0)
    Phi    = dot(g_init, x - s_init)

    iter_no = 0
    start_time = time()
    for t in 1:max_iter
        if time() - start_time > time_limit
            println("Terminating: Time limit of $time_limit seconds exceeded.")
            break
        end
        iter_no += 1
        g = grad_f(Q, c, x)

        # ---------------------------------------------------------
        # Step 2: Find best and worst directions within active set S
        # ---------------------------------------------------------
        gv             = [dot(g, a) for a in atoms]
        best_inner_idx = argmin(gv)
        worst_idx      = argmax(gv)
        
        v_S            = atoms[best_inner_idx]
        v_A            = atoms[worst_idx]
        
        # The maximum possible progress inside S is the Pairwise gap
        g_PW           = gv[worst_idx] - gv[best_inner_idx]

        # Push metrics
        push!(hist_obj, f(Q, c, x))
        push!(hist_gap, Phi)

        if Phi <= tol
            if verbose 
                println("iter=$(lpad(t,4)), [Done], obj=$(round(hist_obj[end], digits=6)), phi=$(round(Phi, digits=6))")
            end
            break
        end

        # ---------------------------------------------------------
        # Step 3: Compare Pairwise inner gap against threshold Phi
        # ---------------------------------------------------------
        if g_PW >= Phi && g_PW > 1e-12
            # ------ 3a. Pairwise Step (Strictly inside S) ------
            step_type = "Pair"
            d       = v_S - v_A
            eta_max = weights[worst_idx] # Max weight we can steal from the worst atom

            if stepsize == :exact_bounded || stepsize == :exact
                η = exact_stepsize_bounded(Q, c, x, d, eta_max)
            elseif stepsize == :diminishing
                η = min(diminishing_stepsize(t), eta_max)
            else
                η = armijo_stepsize(Q, c, x, d, g; eta_init=eta_max)
            end

            # Weight update: Shift weight directly from worst to best atom
            weights[best_inner_idx] += η
            weights[worst_idx]      -= η

        else
            # ------ 3b. FW Step (Global LMO Call) ------
            v_FW = lmo_firstorderlp(g, A, b; iter_limit=lmo_iter, verbose=0)
            g_FW = dot(g, x - v_FW)

            if g_FW >= Phi
                # Take standard FW step
                step_type = "FW  "
                d       = v_FW - x
                eta_max = 1.0

                if stepsize == :exact_bounded || stepsize == :exact
                    η = exact_stepsize_bounded(Q, c, x, d, eta_max)
                elseif stepsize == :diminishing
                    η = min(diminishing_stepsize(t), eta_max)
                else
                    η = armijo_stepsize(Q, c, x, d, g; eta_init=eta_max)
                end

                # Weight update: scale all by (1 - η), add new atom
                weights .*= (1 - η)
                
                # Check if atom already exists to prevent duplicates
                idx = findfirst(a -> norm(a - v_FW) < 1e-10, atoms)
                if idx === nothing
                    push!(atoms, copy(v_FW))
                    push!(weights, η)
                else
                    weights[idx] += η
                end
            else
                # Halve the threshold and take no step
                step_type = "Drop"
                Phi /= 2.0
            end
        end

        if verbose && (t % 100 == 0 || step_type == "Drop")
            println("iter=$(lpad(t,4)), [$step_type], obj=$(round(hist_obj[end], digits=6)), phi=$(round(Phi, digits=6)), g_PW=$(round(g_PW, digits=6))")
        end

        # ---------------------------------------------------------
        # Step 4: Drop atoms with negligible weight
        # ---------------------------------------------------------
        keep    = weights .> 1e-12
        atoms   = atoms[keep]
        weights = weights[keep]

        # Recompute x from active set
        x = sum(weights[i] .* atoms[i] for i in eachindex(atoms))
    end
    
    return x, hist_obj, hist_gap, iter_no
end