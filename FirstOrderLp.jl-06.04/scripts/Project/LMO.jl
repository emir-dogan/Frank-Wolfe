
# ------------------------------------------------------------
# LMO:
#   s = argmin { g'x : A*x <= b, x >= 0 }
#
# FirstOrderLp expects:
#   constraint_matrix * x >= rhs
# so we convert A*x <= b into -A*x >= -b
# ------------------------------------------------------------

function lmo_firstorderlp(g, A, b; iter_limit = 1e+6, verbose = 0)
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
    0,
  )

  term = FirstOrderLp.construct_termination_criteria(
    eps_optimal_absolute = 1e-9,
    eps_optimal_relative = 1e-9,
    iteration_limit = Int32(iter_limit),
  )

  restart = FirstOrderLp.construct_restart_parameters(
    FirstOrderLp.ADAPTIVE_NORMALIZED,
    FirstOrderLp.GAP_OVER_DISTANCE_SQUARED,
    1000,   # fixed restart frequency (unused here, but required)
    0.5,
    0.1,
    0.9,
    0.5,
    false,
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
    FirstOrderLp.AdaptiveStepsizeParams(0.3, 0.6),
  )


  out = FirstOrderLp.optimize(params, lp)
  s = out.primal_solution

  return s
end