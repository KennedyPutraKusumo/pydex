from sir_model import simulate
from pydex.core.designer import Designer
import numpy as np

designer = Designer()
designer.simulate = simulate
designer.ti_controls_candidates = [[100, 1, 0]]
designer.sampling_times_candidates = [np.linspace(0, 10, 31)]
designer.model_parameters = np.random.uniform(
    low=[5, 0.5],
    high=[10, 1.5],
    size=[100, 2]
)

designer.initialize(verbose=2)

designer.solve_cvar_problem(
    designer.cvar_d_opt_criterion,
    beta=0.80,
    optimize_sampling_times=True,
    optimizer="MOSEK",
    package="cvxpy",
    reso=5,
    plot=True,
    write=False,
)
designer.plot_pareto_frontier(write=False)

designer.show_plots()
