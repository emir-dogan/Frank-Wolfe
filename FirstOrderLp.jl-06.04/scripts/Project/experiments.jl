using Random
using LinearAlgebra
using SparseArrays
import FirstOrderLp

include("instance.jl")
include("LMO.jl")
include("search.jl")
include("fw.jl")

function run_experiments()

  ns = [25, 50, 100]
  m_ratios = [0.5, 1, 2]
  rhos = [0.5, 0.95]
  kappas = [10.0, 100.0, 1000.0]
  taus = [0.1, 1, 10.0]

  replications = 9
  max_iter = 1e+5
  tol = 1e-5
  stepsize = :exact_bounded

  csv_file = "experiment_results.csv"
  hist_csv_file = "experiment_history.csv"

  # Initialize CSV with precise headers
  open(csv_file, "w") do io
    println(
      io,
      "n,m,m_n_ratio,rho,kappa,tau,seed,time_sec,termination_reason,final_objective,objective_improvement,frank_wolfe_gap,iterations,l2_norm_diff,true_obj_diff",
    )
  end

  open(hist_csv_file, "w") do io
    println(io, "n,m,m_n_ratio,rho,kappa,tau,seed,iteration,objective,gap")
  end

  seed_counter = 1
  total_configs =
    length(ns) *
    length(m_ratios) *
    length(rhos) *
    length(kappas) *
    length(taus) *
    (replications + 1)

  println("Total experiments to run: ", total_configs)

  for n in ns
    for m_ratio in m_ratios
      m = n * m_ratio
      for rho in rhos
        for kappa in kappas
          for tau in taus
            for rep in 0:replications
              current_seed = seed_counter

              if rep == 0
                Q, c, A, b, x_star = random_qp_known_solution(
                  n, m; kappa = kappa, rho = rho, tau = tau, seed = current_seed,
                )
              else
                # Generate an independent random QP instance
                Q, c, A, b = random_qp(
                  n,
                  m;
                  kappa = kappa,
                  rho = rho,
                  tau = tau,
                  seed = current_seed,
                )
                x_star = nothing
              end

              # Execute the standard Frank-Wolfe optimization 
              start_time = time()
              x_fw, hist_obj, hist_gap, iter_no = frank_wolfe(
                Q,
                c,
                A,
                b;
                max_iter = max_iter,
                tol = tol,
                verbose = true,
                stepsize = stepsize,
              )
              end_time = time()
              elapsed_time = end_time - start_time

              # Gather metrics to log
              final_gap = hist_gap[end]
              term_reason = final_gap <= tol ? "tol" : "max_iter"
              final_obj = f(Q, c, x_fw)       # True final objective at optimal x
              if term_reason == "tol"
                prev_obj = length(hist_obj) > 1 ? hist_obj[end-1] : hist_obj[1]
              else
                prev_obj = length(hist_obj) > 0 ? hist_obj[end] : final_obj
              end
              obj_imp = prev_obj - final_obj

              if rep == 0
                l2_diff = norm(x_fw - x_star)
                true_obj_diff = final_obj - f(Q, c, x_star)
              else
                l2_diff = ""
                true_obj_diff = ""
              end

              open(csv_file, "a") do io
                println(
                  io,
                  "$n,$m,$m_ratio,$rho,$kappa,$tau,$current_seed,$elapsed_time,$term_reason,$final_obj,$obj_imp,$final_gap,$iter_no,$l2_diff,$true_obj_diff",
                )
              end

              if rep == 1
                open(hist_csv_file, "a") do io
                  for t in 1:length(hist_obj)
                    println(
                      io,
                      "$n,$m,$m_ratio,$rho,$kappa,$tau,$current_seed,$t,$(hist_obj[t]),$(hist_gap[t])",
                    )
                  end
                end
              end

              seed_counter += 1
            end
            # Verbosity helper during lengthy runs
            println(
              "Finished configuration: n=$n, m=$m, \t rho=$rho, \t kappa=$kappa, \t tau=$tau",
            )
          end
        end
      end
    end
  end

  println("Experiments completed. Results safely persisted to $csv_file")
end

run_experiments()
