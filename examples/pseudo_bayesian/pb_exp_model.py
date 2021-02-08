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

np.random.seed(123)
n_scr = 100
designer.model_parameters = np.random.multivariate_normal(
    mean=[1, -1],
    cov=np.array(
        [
            [0.2**2, 0.00],
            [0.00, 0.2**2],
        ]
    ),
    size=n_scr,
)
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
    pseudo_bayesian_type=1,
)
designer.print_optimal_candidates(write=False)
designer.plot_optimal_controls()

designer.design_experiment(
    designer.a_opt_criterion,
    write=False,
    package="cvxpy",
    optimizer="MOSEK",
    pseudo_bayesian_type=1,
)
designer.print_optimal_candidates(write=False)
designer.plot_optimal_controls()

designer.design_experiment(
    designer.e_opt_criterion,
    write=False,
    package="cvxpy",
    optimizer="MOSEK",
    pseudo_bayesian_type=1,
)
designer.print_optimal_candidates(write=False)
designer.plot_optimal_controls()

designer.show_plots()
