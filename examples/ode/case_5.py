from pydex.core.designer import Designer
from case_5_model import simulate
import numpy as np

designer = Designer()

designer.simulate = simulate

tic = designer.enumerate_candidates(
    bounds=[
        [273.15, 323.15],
    ],
    levels=[
        11,
    ],
)
designer.ti_controls_candidates = tic

spt = np.array([
    np.linspace(0, 10, 21) for _ in tic
])
designer.sampling_times_candidates = spt

designer._num_steps = 7

# nominal at 1, 1
if True:
    mp = [1, 3, -1, 5]
# uncertain uniform
if False:
    mp = np.random.uniform(
        low=[0, 2, -2, 4],
        high=[2, 4, 0, 6],
        size=[625, 4],
    )
designer.model_parameters = mp

designer.initialize(verbose=2)
designer.reporting_frequency = 2
designer._num_steps = 10

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
