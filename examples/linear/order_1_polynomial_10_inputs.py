from pydex.core.designer import Designer
import numpy as np


"""
Setting     : a non-dynamic experimental system with 4 time-invariant control variables 
              and 1 response.
Problem     : design optimal experiment for a order 1 polynomial.
Solution    : a 2^4 factorial design.
"""

def simulate(ti_controls, model_parameters):
    return np.array([
        # constant
        model_parameters[0] +
        # linear
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        model_parameters[3] * ti_controls[2] +
        model_parameters[4] * ti_controls[3] +
        model_parameters[5] * ti_controls[4] +
        model_parameters[6] * ti_controls[5] +
        model_parameters[7] * ti_controls[6] +
        model_parameters[8] * ti_controls[7] +
        model_parameters[9] * ti_controls[8] +
        model_parameters[10] * ti_controls[9]
    ])

designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.ones(11)
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1]
        for _ in range(10)
    ],
    levels=[
        3
        for _ in range(10)
    ],
)
designer.initialize(verbose=2)
designer.design_experiment(
    designer.d_opt_criterion,
    write=False,
    trim_fim=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls()
designer.show_plots()
