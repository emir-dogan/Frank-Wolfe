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