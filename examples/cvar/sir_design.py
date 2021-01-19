from sir_model import simulate
from pydex.core.designer import Designer
import numpy as np

designer = Designer()
designer.simulate = simulate
designer.ti_controls_candidates = [[100, 1, 0]]
designer.sampling_times_candidates = [np.linspace(0, 10, 501)]
designer.model_parameters = np.random.uniform(
    low=[5, 0.5],
    high=[10, 1.5],
    size=[200, 2]
)

designer.initialize(verbose=2)

designer._num_steps = 6
# designer.design_experiment(
#     designer.d_opt_criterion,
#     optimize_sampling_times=True,
#     optimizer="MOSEK",
#     package="cvxpy",
#     pseudo_bayesian_type=1,
# )
# designer.print_optimal_candidates(write=False, tol=1e-4)
# designer.plot_optimal_sensitivities(interactive=False)
# designer.plot_optimal_predictions()

designer.solve_cvar_problem(
    designer.cvar_d_opt_criterion,
    beta=0.9,
    optimize_sampling_times=True,
    optimizer="MOSEK",
    package="cvxpy",
    pseudo_bayesian_type=1,
)
designer.plot_pareto_frontier(write=True)

designer.show_plots()
