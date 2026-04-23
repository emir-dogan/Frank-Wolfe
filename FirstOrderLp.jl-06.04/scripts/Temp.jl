using Random
using LinearAlgebra
using SparseArrays
import FirstOrderLp

# ------------------------------------------------------------
# Random convex QP:
#   min 0.5*x'Q*x + c'x
#   s.t. A*x <= b, x >= 0
# ------------------------------------------------------------

function random_qp(n::Int, m::Int; seed::Int=1)
    Random.seed!(seed)

    # Make Q positive semidefinite
    M = randn(n, n)
    Q = M' * M + 1e-4 * I(n) 
    Q = sparse(Q)

    c = randn(n) 

    # Build A,b so that x = 0 is feasible
    A = rand(m, n)   
    b = rand(m) .+ 1.0   # positive => A*0 <= b

    return Q, c, A, b
end

# Objective and gradient
function f(Q, c, x)
    return 0.5 * dot(x, Q * x) + dot(c, x)
end

function grad_f(Q, c, x)
    return Q * x + c
end

# ------------------------------------------------------------
# LMO:
#   s = argmin { g'x : A*x <= b, x >= 0 }
#
# FirstOrderLp expects:
#   constraint_matrix * x >= rhs
# so we convert A*x <= b into -A*x >= -b
# ------------------------------------------------------------

function lmo_firstorderlp(g, A, b; iter_limit=1e+6, verbose=0)
    n = length(g)
    m = length(b)

    lb = zeros(n)
    ub = fill(Inf, n)

    # Convert A*x <= b  -->  -A*x >= -b
    K = sparse(-A)
    rhs = -b

    lp = FirstOrderLp.linear_programming_problem(
        lb,
        ub,
        collect(g),
        0.0,
        K,
        collect(rhs),
        0
    )

    term = FirstOrderLp.construct_termination_criteria(
        eps_optimal_absolute=1e-9,
        eps_optimal_relative=1e-9,
        iteration_limit=Int32(iter_limit)
    )

    restart = FirstOrderLp.construct_restart_parameters(
        FirstOrderLp.ADAPTIVE_NORMALIZED,
        FirstOrderLp.GAP_OVER_DISTANCE_SQUARED,
        1000,   # fixed restart frequency (unused here, but required)
        0.5,
        0.1,
        0.9,
        0.5,
        false
    )

    params = FirstOrderLp.PdhgParameters(
        10,          # l_inf_ruiz_iterations
        false,       # l2_norm_rescaling
        1.0,         # pock_chambolle_alpha
        1.0,         # primal_importance
        true,        # scale_invariant_initial_primal_weight
        verbose,     # verbosity
        false,       # record_iteration_stats
        Int32(50),   # termination_evaluation_frequency
        term,
        restart,
        FirstOrderLp.AdaptiveStepsizeParams(0.3, 0.6)
    )


    out = FirstOrderLp.optimize(params, lp)
    s = out.primal_solution

    return s
end

# ------------------------------------------------------------
# Exact line search for quadratic objective along d = s - x
#
# min_{eta in [0,1]} f(x + eta d)
# ------------------------------------------------------------

function exact_stepsize(Q, c, x, d)
    denom = dot(d, Q * d)
    numer = dot(Q * x + c, d)

    if abs(denom) < 1e-12
        return numer < 0 ? 1.0 : 0.0
    else
        η = -numer / denom
        return clamp(η, 0.0, 1.0)
    end
end

# Diminishing stepsize: eta_t = 2 / (t + 2)
function diminishing_stepsize(t::Real)
    return 2.0 / (t + 2)
end

# Armijo backtracking along d = s - x
# Finds eta in (0,1] s.t. f(x + eta*d) <= f(x) + sigma*eta*grad'*d
function armijo_stepsize(Q, c, x, d, g; sigma=1e-4, beta=0.5, eta_init=1.0)
    eta = eta_init
    f0 = f(Q, c, x)
    slope = dot(g, d)
    while eta > 1e-16
        if f(Q, c, x + eta * d) <= f0 + sigma * eta * slope
            return eta
        end
        eta *= beta
    end
    return eta
end

# Exact line search with arbitrary upper bound eta_max
# min_{eta in [0, eta_max]} f(x + eta*d)
function exact_stepsize_bounded(Q, c, x, d, eta_max)
    denom = dot(d, Q * d)
    numer = dot(Q * x + c, d)
    if abs(denom) < 1e-12
        return numer < 0 ? eta_max : 0.0
    else
        return clamp(-numer / denom, 0.0, eta_max)
    end
end

# ------------------------------------------------------------
# Away-step Frank-Wolfe (AFW)
#
# Maintains x as a convex combination of atoms (active set).
# Each iteration chooses the better of:
#   FW step:   d = s - x,  s = LMO(g),  eta in [0, 1]
#   Away step: d = x - v,  v = argmax_{v in S} g.v,  eta in [0, lambda_v/(1-lambda_v)]
# ------------------------------------------------------------

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

            if stepsize == :exact
                η = exact_stepsize_bounded(Q, c, x, d, eta_max)
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

            if stepsize == :exact
                η = exact_stepsize_bounded(Q, c, x, d, eta_max)
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

# ------------------------------------------------------------
# Frank-Wolfe
# ------------------------------------------------------------

function frank_wolfe(Q, c, A, b; max_iter=100, tol=1e-6, verbose=true,
                     stepsize=:diminishing)  # :exact | :diminishing | :armijo 

    # x=0 is feasible by construction
    x = zeros(n)

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

# ------------------------------------------------------------
# Run a toy example
# ------------------------------------------------------------

n = 20   # number of variables
m = 10   # number of constraints

Q, c, A, b = random_qp(n, m; seed=3131)

x_afw, obj_afw, gap_afw, iter_afw = away_step_frank_wolfe(Q, c, A, b; max_iter=1e+6, tol=1e-6 ,verbose=true, stepsize=:exact)
println("\nAFW solution x* =")
println(x_afw)      

println("\nAFW objective value = ", f(Q, c, x_afw))
println("AFW FW gap = ", gap_afw[end])
println("AFW iterations = ", iter_afw)
println("Check AFW feasibility:")       
println("minimum x_i = ", minimum(x_afw))
println("max(Ax - b) = ", maximum(A * x_afw - b))   

# x_star, obj_hist, gap_hist, iter_fw = frank_wolfe(Q, c, A, b; max_iter=1e+6, tol=1e-6, stepsize=:exact, verbose=true)

# println("\nFinal solution x* =")
# println(x_star)

# println("\nFinal objective value = ", f(Q, c, x_star))
# println("Final FW gap = ", gap_hist[end])
# println("FW iterations = ", iter_fw)
# println("Check feasibility:")
# println("minimum x_i = ", minimum(x_star))
# println("max(Ax - b) = ", maximum(A * x_star - b))

