from pydex.core.designer import Designer
import numpy as np


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] +
        model_parameters[1] * np.exp(model_parameters[2] * ti_controls[0])
    ])

designer = Designer()
designer.simulate = simulate

tic = np.mgrid[-1:1:111j]
designer.ti_controls_candidates = tic[:, None]

mp = np.array([1, 2, -3])
designer.model_parameters = mp

designer.initialize()

criterion = designer.a_opt_criterion
designer.design_experiment(criterion, write=False)
designer.print_optimal_candidates()
designer.plot_controls()
