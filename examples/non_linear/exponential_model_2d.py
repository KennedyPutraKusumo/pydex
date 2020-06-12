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

reso = 21j
tic1, tic2 = np.mgrid[-1:1:reso, -1:1:reso]
tic = np.array([tic1.flatten(), tic2.flatten()]).T
designer.ti_controls_candidates = tic

mp = np.array([1, 2, 2, 10, 2])
designer.model_parameters = mp

designer.initialize(verbose=2)

criterion = designer.a_opt_criterion
designer.design_experiment(criterion, write=False)
designer.print_optimal_candidates()
designer.plot_controls(non_opt_candidates=True)
designer.show_plots()
