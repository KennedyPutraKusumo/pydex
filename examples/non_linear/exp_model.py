import numpy as np

from pydex.core.designer import Designer


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] * np.exp(model_parameters[1] * ti_controls[0])
    ])

designer = Designer()
designer.simulate = simulate

reso = 21j
tic = np.mgrid[0:1:reso]
designer.ti_controls_candidates = np.array([tic]).T

designer.model_parameters = [1, -1]

designer.initialize(verbose=2)

""" 
Pseudo-bayesian type do not really matter in this case because only a single model 
parameter is involved i.e, information is a scalar, all criterion becomes equivalent to 
the information matrix itself.
"""

designer.design_experiment(
    designer.d_opt_criterion,
    write=False,
    package="cvxpy",
    optimizer="MOSEK",
)
designer.print_optimal_candidates()
designer.plot_optimal_controls()
designer.show_plots()
