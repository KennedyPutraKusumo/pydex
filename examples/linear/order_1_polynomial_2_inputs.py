from pydex.core.designer import Designer
import numpy as np


"""
Setting     : a non-dynamic experimental system with 2 time-invariant control variables 
              and 1 response.
Problem     : design optimal experiment for a order 1 polynomial.
Solution    : a 2^2 factorial design, criterion does not affect design.
"""

def simulate(ti_controls, model_parameters):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1]
    ])

designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.ones(3)  # values won't affect design, but still needed
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=np.array([
        [-1, 1],
        [-1, 1],
    ]),
    levels=np.array([
        11,
        11,
    ])
)

designer.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed
designer.design_experiment(
    designer.e_opt_criterion,
    write=False,
)
# designer.print_optimal_candidates_2()
# designer.plot_optimal_controls_2(non_opt_candidates=True)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True, write=False, markersize=3)
designer.show_plots()
