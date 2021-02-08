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
        model_parameters[4] * ti_controls[3]
    ])

designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.ones(5)
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
    ],
    levels=[
        11,
        11,
        11,
        11,
    ],
)
designer.initialize(verbose=2)
designer.design_experiment(
    designer.d_opt_criterion,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls()
designer.show_plots()
