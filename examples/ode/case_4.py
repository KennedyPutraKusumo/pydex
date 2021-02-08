from pydex.core.designer import Designer
from case_4_model import simulate
import numpy as np

designer = Designer()

designer.simulate = simulate
tic = designer.enumerate_candidates(
    bounds=[
        [0, 0.3],
    ],
    levels=[
        11,
    ],
)
designer.ti_controls_candidates = tic
spt = np.array([np.linspace(0, 10, 101) for _ in tic])
designer.sampling_times_candidates = spt

# nominal at 1, 1
if True:
    mp = [1, 1]
# uncertain uniform
if False:
    mp = np.random.uniform(
        low=[0.5, 0.5],
        high=[1.5, 1.5],
        size=[100, 2],
    )
designer.model_parameters = mp

designer.initialize(verbose=2)
designer.sens_report_freq = 2

""" Pseudo-Bayesian Type 1 D-opt Design """
pkg = "cvxpy"
opt = "MOSEK"
if True:
    designer.design_experiment(
        designer.d_opt_criterion,
        optimize_sampling_times=True,
        package=pkg,
        optimizer=opt,
        pseudo_bayesian_type=1,
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_predictions()
    designer.plot_optimal_sensitivities()

    designer.design_experiment(
        designer.a_opt_criterion,
        optimize_sampling_times=True,
        package=pkg,
        optimizer=opt,
        pseudo_bayesian_type=1,
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_predictions()
    designer.plot_optimal_sensitivities()

    designer.design_experiment(
        designer.e_opt_criterion,
        optimize_sampling_times=True,
        package=pkg,
        optimizer=opt,
        pseudo_bayesian_type=1,
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_predictions()
    designer.plot_optimal_sensitivities()

""" CVaR - Mean Pareto Efficient Designs """
if False:
    designer.solve_cvar_problem(
        designer.cvar_d_opt_criterion,
        beta=0.90,
        package=pkg,
        optimizer=opt,
        optimize_sampling_times=True,
    )
    designer.plot_pareto_frontier(write=False)

designer.show_plots()
