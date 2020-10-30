from pydex.core.designer import Designer
import numpy as np


"""
Setting     : a non-dynamic experimental system with 3 time-invariant control variables 
              and 1 response.
Problem     : design optimal experiment for a order 1 polynomial.
Solution    : a 2^3 factorial design, criterion does not affect design.
"""

def simulate(ti_controls, model_parameters):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        model_parameters[3] * ti_controls[2]
    ])

designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.ones(4)  # values won't affect design, but still needed
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1],
        [-1, 1],
        [-1, 1],
    ],
    levels=[
        11,
        11,
        11,
    ],
)


designer.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed
designer.design_experiment(
    designer.d_opt_criterion,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls()
designer.show_plots()
