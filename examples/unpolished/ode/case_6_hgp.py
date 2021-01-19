from case_6_hgp_model import simulate_hgp
from pydex.core.designer import Designer
import numpy as np


designer = Designer()
designer.simulate = simulate_hgp

tic = designer.enumerate_candidates(
    bounds=[
        [16, 20],
        [10, 200],
        [0.1, 10],
    ],
    levels=[
        1,
        1,
        5
    ]
)
designer.ti_controls_candidates = tic
designer.sampling_times_candidates = [
    np.linspace(0, 24 * 14, 14) for _ in tic
]
# mp = [
#     3.1,            # estimated gas in place - taken from wikipedia page of ghawar field on 2020-12-25
#     1000,  # viscosity in MPa.hr - converted from 1.107x10^(-5) Pa.s
# ]
mp = [
    3.1,
    # estimated gas in place - taken from wikipedia page of ghawar field on 2020-12-25
    # 1000,  # viscosity in MPa.hr - converted from 1.107x10^(-5) Pa.s
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
designer.plot_optimal_sensitivities(interactive=True)
designer.plot_optimal_predictions()
designer.show_plots()
