# ------------------------------------------------------------
# Random convex QP:
#   min 0.5*x'Q*x + c'x
#   s.t. A*x <= b, x >= 0
# ------------------------------------------------------------

function random_qp(
  n::Int,
  m::Int;
  kappa::Float64 = 10.0,
  rho::Float64 = 0.1,
  tau::Float64 = 1.0,
  seed::Int = 1,
)
  Random.seed!(seed)

  # Control of Objective Function
  # 1. The quadratic term Q
  G = randn(n, n)
  F = svd(G)
  U_mat = F.U * F.Vt

  if n > 1
    lambda = [exp((i - 1) / (n - 1) * log(kappa)) for i in 1:n]
  else
    lambda = [1.0]
  end

  Q_dense = U_mat' * Diagonal(lambda) * U_mat
  Q_dense = (Q_dense + Q_dense') / 2 # Ensure symmetry
  Q = sparse(Q_dense)

  # 2. Linear term c
  c = randn(n)

  # Geometry of Polytope
  # Sparsity and Randomization
  A = sprand(m, n, rho)  #uniform distribution in [0,1] with probability 1-rho, 0 with probability rho
  for j in 1:n
      if nnz(A[:, j]) == 0
          # Pick a random row and assign a random uniform value (0, 1]
          random_row = rand(1:m)
          A[random_row, j] = rand()
      end
  end
  # Constraint tightness
  row_norms = sum(A, dims = 2)
  u = 0.75 .+ 0.5 .* rand(m) # uniform distribution in [0.75, 1.25]
  b = vec(tau .* row_norms .* u)

  return Q, c, A, b
end

function random_qp_known_solution(n::Int, m::Int; kappa::Float64=10.0, rho::Float64=0.1, tau::Float64=1.0, seed::Int=1)
    Random.seed!(seed)

    G = randn(n, n)
    F = svd(G)
    U_mat = F.U * F.Vt
    
    if n > 1
        lambda = [exp((i - 1) / (n - 1) * log(kappa)) for i in 1:n]
    else
        lambda = [1.0]
    end
    
    Q_dense = U_mat' * Diagonal(lambda) * U_mat
    Q_dense = (Q_dense + Q_dense') / 2
    Q = sparse(Q_dense)

    A = sprand(m, n, rho)   
    
    row_norms = sum(A, dims=2)
    u = 0.75 .+ 0.5 .* rand(m)
    b = vec(tau .* row_norms .* u)

    # Validation: target optimal point
    x_0 = rand(n)
    Ax0 = A * x_0
    
    valid_mask = b .> 0
    if any(valid_mask)
        gamma = maximum(Ax0[valid_mask] ./ b[valid_mask]) + 0.05
    else
        gamma = 1.0
    end
    
    x_star = x_0 ./ gamma
    
    c = -Q * x_star

    return Q, c, A, b, x_star
end