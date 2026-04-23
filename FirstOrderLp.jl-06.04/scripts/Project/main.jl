import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using LinearAlgebra
using SparseArrays
import FirstOrderLp

include("instance.jl")
include("LMO.jl")
include("search.jl")
include("fw.jl")
include("fw_away.jl")
include("fw_blended.jl")
include("fw_bpcg.jl")
include("Gurobi.jl")

max_iter = 1e+6
time_limit = 300.0
tol = 1e-6
verbose = true
stepsize = :exact_bounded # :exact | :exact_bounded | :diminishing | :armijo #bounded away step için necessary, düzde bir şeyi değiştirmiyor
# ------------------------------------------------------------
# Run a toy example
# ------------------------------------------------------------

n = 500   # number of variables
m = 500   # number of constraints

Q, c, A, b = random_qp(n, m; seed=3131)
println("Generated random QP instance with n = $n variables and m = $m constraints.")

start_time = time()
x_gurobi, obj_gurobi = solve_with_gurobi(Q, c, A, b)
elapsed_time = time() - start_time
println("\nGurobi completed in $(round(elapsed_time, digits=2)) seconds.")
println("\nGurobi solution x* =")
# println(x_gurobi)   
println("\nGurobi objective value = ", f(Q, c, x_gurobi))
println("Check Gurobi feasibility:")
println("minimum x_i = ", minimum(x_gurobi))
println("max(Ax - b) = ", maximum(A * x_gurobi - b))

# start_time = time()
# x_afw, obj_afw, gap_afw, iter_afw = away_step_frank_wolfe(Q, c, A, b;time_limit=time_limit, max_iter=max_iter, tol=tol ,verbose=verbose, stepsize=stepsize)
# elapsed_time = time() - start_time
# println("\nAway-Step FW completed in $(round(elapsed_time, digits=2)) seconds.")
# println("\nAFW solution x* =")
# println(x_afw)      

# println("\nAFW objective value = ", f(Q, c, x_afw))
# println("AFW FW gap = ", gap_afw[end])
# println("AFW iterations = ", iter_afw)
# println("Check AFW feasibility:")       
# println("minimum x_i = ", minimum(x_afw))
# println("max(Ax - b) = ", maximum(A * x_afw - b))   

start_time = time()
x_bpcg, obj_bpcg, gap_bpcg, iter_bpcg = blended_pairwise_frank_wolfe(Q, c, A, b;time_limit=time_limit, max_iter=max_iter, tol=tol ,verbose=verbose, stepsize=stepsize)
elapsed_time = time() - start_time
println("\nBlended Pairwise FW completed in $(round(elapsed_time, digits=2)) seconds.")
println("\nBlended Pairwise FW solution x* =")
# println(x_bpcg)

println("\nBlended Pairwise FW objective value = ", f(Q, c, x_bpcg))
println("Blended Pairwise FW gap = ", gap_bpcg[end])
println("Blended Pairwise FW iterations = ", iter_bpcg)
println("Check Blended Pairwise FW feasibility:")
println("minimum x_i = ", minimum(x_bpcg))
println("max(Ax - b) = ", maximum(A * x_bpcg - b))

# start_time = time()
# x_blended, obj_blended, gap_blended, iter_blended = blended_frank_wolfe(Q, c, A, b;time_limit=time_limit, max_iter=max_iter, tol=tol ,verbose=verbose, stepsize=stepsize)
# elapsed_time = time() - start_time
# println("\nBlended FW completed in $(round(elapsed_time, digits=2)) seconds.")
# println("\nBlended FW solution x* =")
# println(x_blended)

# println("\nBlended FW objective value = ", f(Q, c, x_blended))
# println("Blended FW gap = ", gap_blended[end])
# println("Blended FW iterations = ", iter_blended)
# println("Check Blended FW feasibility:")    
# println("minimum x_i = ", minimum(x_blended))
# println("max(Ax - b) = ", maximum(A * x_blended - b))

# start_time = time()
# x_star, obj_hist, gap_hist, iter_fw = frank_wolfe(Q, c, A, b;time_limit=time_limit, max_iter=max_iter, tol=tol, stepsize=stepsize, verbose=verbose)
# elapsed_time = time() - start_time
# println("\nFrank-Wolfe completed in $(round(elapsed_time, digits=2)) seconds.")
# println("\nFinal solution x* =")
# println(x_star)

# println("\nFinal objective value = ", f(Q, c, x_star))
# println("Final FW gap = ", gap_hist[end])
# println("FW iterations = ", iter_fw)
# println("Check feasibility:")
# println("minimum x_i = ", minimum(x_star))
# println("max(Ax - b) = ", maximum(A * x_star - b))