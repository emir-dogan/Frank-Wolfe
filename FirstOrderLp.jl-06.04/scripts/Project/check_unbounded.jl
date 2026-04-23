using Random
using SparseArrays

n = 10
m = 10
rho = 0.05
seed = 12

Random.seed!(seed)

# A is generated exactly as it is in your random_qp methods
A = sprand(m, n, rho)

println("Matrix A (10x10) with rho=0.05, seed=12\n")
println("Total non-zero elements: ", nnz(A))
println("\nHere is exactly how Matrix A populated (notice the structural zeros):")
display(Matrix(A))

zero_cols = [j for j in 1:n if sum(A[:, j]) == 0]
println("\nColumns ENTIRELY empty across all rows: ", zero_cols)

if !isempty(zero_cols)
    println("\n!!! UNBOUNDED CONFIRMED !!!")
    println("Because variables ", zero_cols, " have absolutely ZERO boundaries assigned to them inside A, they are free to increase to +Infinity mathematically. The region geometry is absolutely unbounded along these directional rays.")
else
    println("\nAll columns managed to receive at least one bounding value.")
end
