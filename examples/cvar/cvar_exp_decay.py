import numpy as np
from pydex.core.designer import Designer


def simulate(ti_controls, model_parameters):
    return np.array([
        np.exp(model_parameters[0] * ti_controls[0])
    ])

designer = Designer()
designer.simulate = simulate

reso = 21j
tic = np.mgrid[0:1:reso]
designer.ti_controls_candidates = np.array([tic]).T

np.random.seed(123)
n_scr = 10
designer.model_parameters = np.random.normal(loc=-1, scale=0.50, size=(n_scr, 1))

designer.start_logging()
designer.initialize(verbose=1)

""" 
Pseudo-bayesian type do not really matter in this case because only a single model 
parameter is involved i.e, information is a scalar, all criterion becomes equivalent to 
the information matrix itself.
"""

designer.solve_cvar_problem(
    designer.cvar_d_opt_criterion,
    beta=0.89,
    reso=5,
    plot=True,
)
designer.plot_pareto_frontier(write=True)
designer.stop_logging()
designer.show_plots()
