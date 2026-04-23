# ------------------------------------------------------------
# Frank-Wolfe
# ------------------------------------------------------------

# Objective and gradient
function f(Q, c, x)
    return 0.5 * dot(x, Q * x) + dot(c, x)
end

function grad_f(Q, c, x)
    return Q * x + c
end

function frank_wolfe(Q, c, A, b; max_iter=100, tol=1e-6, verbose=true,
                     stepsize=:exact)  # :exact | :diminishing | :armijo 

    # x=0 is feasible by construction
    x = zeros(size(A, 2))

    hist_obj = Float64[]
    hist_gap = Float64[]
    iter_no = 0
    for t in 1:max_iter
        iter_no +=1
        g = grad_f(Q, c, x)
        s = lmo_firstorderlp(g, A, b; iter_limit=3000, verbose=0)

        d = s - x
        gap = -dot(g, d)   # Frank-Wolfe gap

        rel_obj_change = isempty(hist_obj) ? Inf : abs(hist_obj[end] - f(Q, c, x)) / max(abs(hist_obj[end]), 1e-8)
        rel_gap_change = isempty(hist_gap) ? Inf : abs(hist_gap[end] - gap) / max(abs(hist_gap[1]), 1e-8)
        push!(hist_obj, f(Q, c, x))
        push!(hist_gap, gap)

        if verbose && (t % 100 == 0 || gap <= tol)
            println("iter = $(t), obj = $(round(hist_obj[end], digits=6)), gap = $(round(gap, digits=6))")
        end

        if gap <= tol
            println("Terminating: Frank-Wolfe gap $gap <= tol $tol")
            break
        end

        if stepsize == :exact
            η = exact_stepsize(Q, c, x, d)
        elseif stepsize == :exact_bounded
                η = exact_stepsize_bounded(Q, c, x, d, 1.0)
        elseif stepsize == :diminishing
            η = diminishing_stepsize(t)
        elseif stepsize == :armijo
            η = armijo_stepsize(Q, c, x, d, g)
        else
            error("Unknown stepsize: $stepsize. Choose :exact, :diminishing, or :armijo.")
        end
        x = x + η * d
    end
    return x, hist_obj, hist_gap , iter_no
end