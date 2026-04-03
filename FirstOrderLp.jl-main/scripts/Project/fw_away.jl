# ------------------------------------------------------------
# Away-step Frank-Wolfe (AFW)
#
# Maintains x as a convex combination of atoms (active set).
# Each iteration chooses the better of:
#   FW step:   d = s - x,  s = LMO(g),  eta in [0, 1]
#   Away step: d = x - v,  v = argmax_{v in S} g.v,  eta in [0, lambda_v/(1-lambda_v)]
# ------------------------------------------------------------

# Objective and gradient
function f(Q, c, x)
    return 0.5 * dot(x, Q * x) + dot(c, x)
end

function grad_f(Q, c, x)
    return Q * x + c
end

function away_step_frank_wolfe(Q, c, A, b; max_iter=100, tol=1e-6, verbose=true,
                                stepsize=:exact)  # :exact | :diminishing | :armijo
    max_iter = isinf(max_iter) ? typemax(Int) : Int(max_iter)
    n = length(c)

    # Active set: x = sum_i weights[i] * atoms[i]
    atoms   = [zeros(n)]
    weights = [1.0]
    x       = zeros(n)

    hist_obj = Float64[]
    hist_gap = Float64[]

    iter_no = 0
    for t in 1:max_iter
        iter_no += 1
        g = grad_f(Q, c, x)

        # FW atom and gap: s = argmin_{s in C} g.s,  fw_gap = g.(x - s) >= 0
        s      = lmo_firstorderlp(g, A, b; iter_limit=3000, verbose=0)
        fw_gap = dot(g, x - s)

        # Away atom and gap: v = argmax_{v in S} g.v,  away_gap = g.(v - x) >= 0
        gv        = [dot(g, a) for a in atoms]
        away_idx  = argmax(gv)
        v         = atoms[away_idx]
        away_gap  = gv[away_idx] - dot(g, x)   # >= 0 since x is a convex combo of atoms

        push!(hist_obj, f(Q, c, x))
        push!(hist_gap, fw_gap)

        if verbose
            step_type = fw_gap >= away_gap ? "FW  " : "Away"
            println("iter=$(lpad(t,4)), [$step_type], obj=$(round(hist_obj[end], digits=6)), fw_gap=$(round(fw_gap, digits=6))")
        end

        if fw_gap <= tol
            break
        end

        if fw_gap >= away_gap
            # ------ FW step ------
            d       = s - x
            eta_max = 1.0

            if stepsize == :exact_bounded
                η = exact_stepsize_bounded(Q, c, x, d, eta_max)
            elseif stepsize == :exact
                η = exact_stepsize(Q, c, x, d)
            elseif stepsize == :diminishing
                η = min(diminishing_stepsize(t), eta_max)
            else
                η = armijo_stepsize(Q, c, x, d, g; eta_init=eta_max)
            end

            # Weight update: scale all by (1 - η), add s
            weights .*= (1 - η)
            idx = findfirst(a -> norm(a - s) < 1e-10, atoms)
            if idx === nothing
                push!(atoms, copy(s))
                push!(weights, η)
            else
                weights[idx] += η
            end
        else
            # ------ Away step ------
            d       = x - v
            λ_v     = weights[away_idx]
            eta_max = λ_v / (1 - λ_v + 1e-15)

            if stepsize == :exact_bounded
                η = exact_stepsize_bounded(Q, c, x, d, eta_max)
            elseif stepsize == :exact
                η = exact_stepsize(Q, c, x, d)
            elseif stepsize == :diminishing
                η = min(diminishing_stepsize(t), eta_max)
            else
                η = armijo_stepsize(Q, c, x, d, g; eta_init=min(eta_max, 1.0))
            end

            # Weight update: scale all by (1 + η), reduce v's weight
            weights        .*= (1 + η)
            weights[away_idx] -= η
        end

        # Drop atoms with negligible weight
        keep    = weights .> 1e-12
        atoms   = atoms[keep]
        weights = weights[keep]

        # Recompute x from active set
        x = sum(weights[i] .* atoms[i] for i in eachindex(atoms))
    end
    return x, hist_obj, hist_gap, iter_no
end