import numpy as np
from matplotlib import pyplot as plt

from pydex.core.designer import Designer

""" 
Setting: a non-dynamic experimental system with 2 time-invariant control variables and 1 response.
Problem: develop a statistical model using an exponential regression model. Use integer values for the exponential
         growth rates between (and including) -1 and 1. 
Solution: a standard 3^2 factorial design
"""


def simulate(ti_controls, tv_controls, model_parameters, sampling_times):
    return np.array([
        model_parameters[0] +

        model_parameters[1] * np.exp(-1 * ti_controls[0]) +
        model_parameters[2] * np.exp(1 * ti_controls[0]) +

        model_parameters[3] * np.exp(-1 * ti_controls[1]) +
        model_parameters[4] * np.exp(1 * ti_controls[1])
    ])


designer_1 = Designer()
designer_1.simulate = simulate

tic_1, tic_2 = np.mgrid[-1:1:11j, -1:1:11j]
tic_1 = tic_1.flatten()
tic_2 = tic_2.flatten()
designer_1.ti_controls_candidates = np.array([tic_1, tic_2]).T

designer_1.model_parameters = np.ones(5)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

designer_1.design_experiment(designer_1.d_opt_criterion, write=False)
designer_1.print_optimal_candidates()
designer_1.plot_current_design()

fig1 = plt.figure()
axes1 = fig1.add_subplot(111)
axes1.scatter(designer_1.ti_controls_candidates[:, 0],
              designer_1.ti_controls_candidates[:, 1],
              s=np.round(designer_1.efforts * 1000, decimals=2))
plt.show()
