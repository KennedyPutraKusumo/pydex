import numpy as np

from pydex.core.designer import Designer

""" 
Setting: a non-dynamic experimental system with 2 time-invariant control variables and 1 response.
Problem: develop a statistical model using an exponential regression model. Use integer values for the exponential
         growth rates between (and including) -2 and 2. 
Solution: non-standard design
"""


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] +

        model_parameters[1] * np.exp(-1 * ti_controls[0]) +
        model_parameters[2] * np.exp(1 * ti_controls[0]) +
        model_parameters[3] * np.exp(-2 * ti_controls[0]) +
        model_parameters[4] * np.exp(2 * ti_controls[0]) +

        model_parameters[5] * np.exp(-1 * ti_controls[1]) +
        model_parameters[6] * np.exp(1 * ti_controls[1]) +
        model_parameters[7] * np.exp(-2 * ti_controls[1]) +
        model_parameters[8] * np.exp(2 * ti_controls[1])
    ])


designer_1 = Designer()
designer_1.simulate = simulate

reso = 11
tic = designer_1.create_grid([[-1, 1], [-1, 1]], [reso, reso])
designer_1.ti_controls_candidates = tic

designer_1.model_parameters = np.ones(9)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

designer_1.design_experiment(designer_1.d_opt_criterion, write=False)

designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_controls()
