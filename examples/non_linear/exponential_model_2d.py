from pydex.core.designer import Designer
import numpy as np


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] +
        model_parameters[1] * np.exp(model_parameters[2] * ti_controls[0]) +
        model_parameters[3] * np.exp(model_parameters[4] * ti_controls[1])
    ])

designer = Designer()
designer.simulate = simulate
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1],
        [-1, 1],
    ],
    levels=[
        11,
        11,
    ],
)
designer.ti_controls_names = ["$x_1$", "$x_2$"]

mp = np.array([1, 2, 2, 10, 2])

designer._num_steps = 30
designer.model_parameters = mp

designer.initialize(verbose=2)

criterion = designer.d_opt_criterion
designer.design_experiment(criterion, write=False)
designer.print_optimal_candidates(write=False)
designer.plot_optimal_controls(write=False, title=True, non_opt_candidates=True, tol=1e-3)

criterion = designer.a_opt_criterion
designer.design_experiment(criterion, write=False)
designer.print_optimal_candidates(write=False)
designer.plot_optimal_controls(write=False, title=True, non_opt_candidates=True, tol=1e-3)

criterion = designer.e_opt_criterion
designer.design_experiment(criterion, write=False, optimizer="MOSEK")
designer.print_optimal_candidates(tol=3e-3, write=False)
designer.plot_optimal_controls(write=False, title=True, non_opt_candidates=True, tol=1e-3)
designer.show_plots()
