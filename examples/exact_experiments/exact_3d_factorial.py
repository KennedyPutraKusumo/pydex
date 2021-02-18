from pydex.core.designer import Designer
import numpy as np


""" 
Setting: a non-dynamic experimental system with 3 time-invariant control variables and 
1 response.
Problem: design optimal experiment for a order 2 polynomial, with complete interaction
Solution: a full 2^3 factorial design (2 level)
"""


def simulate(ti_controls, model_parameters):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        model_parameters[3] * ti_controls[2] +
        # 2-interaction term
        model_parameters[4] * ti_controls[0] * ti_controls[1] +
        model_parameters[5] * ti_controls[0] * ti_controls[2] +
        model_parameters[6] * ti_controls[1] * ti_controls[2] +
        # 3-interaction term
        model_parameters[7] * ti_controls[0] * ti_controls[1] * ti_controls[2]
    ])


designer_1 = Designer()
designer_1.simulate = simulate

reso = 5
tic = designer_1.create_grid(
    bounds=np.array([(-1, 1), (-1, 1), (-1, 1)]),
    levels=np.array([reso, reso, reso])
)
designer_1.ti_controls_candidates = tic

designer_1.model_parameters = np.ones(8)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

designer_1.design_experiment(
    designer_1.d_opt_criterion,
    n_exp=10,
    optimizer="MOSEK"
)

designer_1.print_optimal_candidates()
designer_1.plot_optimal_controls()
designer_1.show_plots()
