from pydex.core.designer import Designer
import numpy as np


""" 
Setting: a non-dynamic experimental system with 2 time-invariant control variables and 1 response
Problem: design optimal experiment for a order 2 polynomial, with complete interaction
Solution: a full 3^2 factorial design (3 level)
"""
def simulate(ti_controls, model_parameters):
    inner_designer = Designer()
    return_sensitivities = inner_designer.detect_sensitivity_analysis_function()
    res = np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0]                    +
        model_parameters[2] * ti_controls[1]                    +
        # interaction term
        model_parameters[3] * ti_controls[0] * ti_controls[1]   +
        # squared term
        model_parameters[4] * ti_controls[0] ** 2               +
        model_parameters[5] * ti_controls[1] ** 2
    ])
    if return_sensitivities:
        sens = np.array([
            [
                [1, ti_controls[0], ti_controls[1], ti_controls[0] * ti_controls[1], ti_controls[0] ** 2 , ti_controls[1] ** 2],
            ],
        ])
        return res, sens
    else:
        return res

designer_1 = Designer()
designer_1.use_finite_difference = False
designer_1.simulate = simulate

reso = 21
tic = designer_1.create_grid(
    bounds=[(-1, 1), (-1, 1)],
    levels=[reso, reso]
)
designer_1.ti_controls_candidates = tic

designer_1.model_parameters = np.ones(6)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

designer_1.design_experiment(
    designer_1.d_opt_criterion,
    write=False,
    n_exp=6,
    discrete_design_solver="OA",
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_controls(alpha=0.3, non_opt_candidates=True)
designer_1.show_plots()
