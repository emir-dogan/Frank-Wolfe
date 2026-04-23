using JuMP, Gurobi

function solve_with_gurobi(Q, c, A, b,time_limit=300)
    model = Model(Gurobi.Optimizer)
    # set_silent(model)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    set_optimizer_attribute(model, "OptimalityTol", 1e-6)
    set_optimizer_attribute(model, "TimeLimit", time_limit)  

    @variable(model, x[1:size(A,2)] >= 0)
    @objective(model, Min, 0.5 * dot(x, Q * x) + dot(c, x))
    @constraint(model, A * x .<= b)

    optimize!(model)

    return value.(x), objective_value(model)
end