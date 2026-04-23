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

# Exact line search with arbitrary upper bound eta_max
# min_{eta in [0, eta_max]} f(x + eta*d)
function exact_stepsize_bounded(Q, c, x, d, eta_max = 1)
  denom = dot(d, Q * d)
  numer = dot(Q * x + c, d)
  if abs(denom) < 1e-12
    return numer < 0 ? eta_max : 0.0
  else
    return clamp(-numer / denom, 0.0, eta_max)
  end
end

# Diminishing stepsize: eta_t = 2 / (t + 2)
function diminishing_stepsize(t::Real)
  return 2.0 / (t + 2)
end

# Armijo backtracking along d = s - x
# Finds eta in (0,1] s.t. f(x + eta*d) <= f(x) + sigma*eta*grad'*d
function armijo_stepsize(
  Q,
  c,
  x,
  d,
  g;
  sigma = 1e-4,
  beta = 0.5,
  eta_init = 1.0,
)
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

