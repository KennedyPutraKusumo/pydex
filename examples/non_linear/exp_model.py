import numpy as np
import casadi

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

designer.model_parameters = [5.5, -5]

designer._norm_sens_by_params = False
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

designer.error_cov = np.diag([0.1])
n_exps = [2, 3, 4, 6, 10, 20, 40]
seed = 123456
for n_exp in n_exps:
    designer.apportion(n_exp)
    mp_bounds = np.array([
        [0, 10],
        [-40, 0],
    ])
    designer.insilico_bayesian_inference(
        n_walkers=32,
        n_steps=5000,
        burn_in=100,
        bounds=mp_bounds,
    )
    designer.plot_bayesian_inference_samples(
        contours=True,
        density=False,
        bounds=mp_bounds,
        title=f"{n_exp} Experiments, Seed: {seed}",
        plot_fim_confidence=True,
    )

designer.show_plots()
