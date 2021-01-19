from case_6_hgp_tvc_model import simulate_hgp
from pydex.core.designer import Designer
import numpy as np


designer = Designer()
designer.simulate = simulate_hgp

tic, tvc = designer.enumerate_candidates(
    bounds=[
        [100, 200],
        [1, 10],
        [21.980, 22],
    ],
    levels=[
        1,
        1,
        5,
    ],
    switching_times=[
        None,
        None,
        np.linspace(0, 1, 5)[:-1],
    ],
)
designer.ti_controls_candidates = tic
designer.tv_controls_candidates = tvc
designer.sampling_times_candidates = [
    np.linspace(0, 24 * 14, 11) for _ in tic
]
mp = [
    3.1,    # estimated gas in place in 10^12 m3 - taken from wikipedia page of Ghawar field on 2020-12-25
]
designer.model_parameters = mp

designer.initialize(verbose=2)

designer.design_experiment(
    designer.d_opt_criterion,
    optimizer="MOSEK",
    package="cvxpy",
    optimize_sampling_times=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_predictions()
designer.plot_optimal_sensitivities()
designer.plot_optimal_predictions()
designer.show_plots()
