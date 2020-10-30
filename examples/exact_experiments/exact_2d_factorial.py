from pydex.core.designer import Designer
import numpy as np


""" 
Setting: a non-dynamic experimental system with 2 time-invariant control variables and 1 response
Problem: design optimal experiment for a order 1 polynomial, with complete interaction
Solution: a full 2^2 factorial design (2 level)
"""
def simulate(ti_controls, model_parameters):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0]                    +
        model_parameters[2] * ti_controls[1]                    +
        # interaction term
        model_parameters[3] * ti_controls[0] * ti_controls[1]
    ])

designer_1 = Designer()
designer_1.simulate = simulate

reso = 7
tic = designer_1.create_grid(
    bounds=[(-1, 1), (-1, 1)],
    levels=[reso, reso]
)
designer_1.ti_controls_candidates = tic

designer_1.model_parameters = np.ones(4)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

designer_1.design_experiment(designer_1.d_opt_criterion, n_exp=6, write=False)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_controls(alpha=0.3, non_opt_candidates=True)
designer_1.show_plots()
