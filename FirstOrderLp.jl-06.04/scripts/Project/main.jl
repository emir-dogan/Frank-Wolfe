using Random
using LinearAlgebra
using SparseArrays
import FirstOrderLp

include("instance.jl")
include("LMO.jl")
include("search.jl")
include("fw.jl")
include("fw_away.jl")

max_iter = 1e+6
tol = 1e-6
verbose = true
stepsize = :exact_bounded # :exact | :exact_bounded | :diminishing | :armijo #bounded away step için necessary, düzde bir şeyi değiştirmiyor
# ------------------------------------------------------------
# Run a toy example
# ------------------------------------------------------------

n = 8   # number of variables
m = 6   # number of constraints

Q, c, A, b = random_qp(n, m; seed=3131)

x_afw, obj_afw, gap_afw, iter_afw = away_step_frank_wolfe(Q, c, A, b; max_iter=max_iter, tol=tol ,verbose=verbose, stepsize=stepsize)
println("\nAFW solution x* =")
println(x_afw)      

println("\nAFW objective value = ", f(Q, c, x_afw))
println("AFW FW gap = ", gap_afw[end])
println("AFW iterations = ", iter_afw)
println("Check AFW feasibility:")       
println("minimum x_i = ", minimum(x_afw))
println("max(Ax - b) = ", maximum(A * x_afw - b))   

# x_star, obj_hist, gap_hist, iter_fw = frank_wolfe(Q, c, A, b; max_iter=max_iter, tol=tol, stepsize=stepsize, verbose=verbose)

# println("\nFinal solution x* =")
# println(x_star)

# println("\nFinal objective value = ", f(Q, c, x_star))
# println("Final FW gap = ", gap_hist[end])
# println("FW iterations = ", iter_fw)
# println("Check feasibility:")
# println("minimum x_i = ", minimum(x_star))
# println("max(Ax - b) = ", maximum(A * x_star - b))