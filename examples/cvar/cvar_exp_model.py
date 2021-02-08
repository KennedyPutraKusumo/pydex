import numpy as np
from pydex.core.designer import Designer


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] * np.exp(model_parameters[1] * ti_controls[0])
    ])

designer = Designer()
designer.simulate = simulate

reso = 21j
tic = np.mgrid[0:0.5:reso]
designer.ti_controls_candidates = np.array([tic]).T

np.random.seed(123)
n_scr = 200
designer.model_parameters = np.random.multivariate_normal(
    mean=[10, -5],
    cov=np.array(
        [
            [2**2, 0.00],
            [0.00, 1.5**2],
        ]
    ),
    size=n_scr,
)
designer._num_steps = 6

designer.start_logging()
designer.initialize(verbose=2)

designer.solve_cvar_problem(
    designer.cvar_d_opt_criterion,
    beta=0.75,
    reso=5,
    plot=True,
)
designer.plot_pareto_frontier()
designer.stop_logging()
designer.show_plots()
