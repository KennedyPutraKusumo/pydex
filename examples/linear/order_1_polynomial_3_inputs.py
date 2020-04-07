from core.designer import Designer
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

""" 
Setting: a non-dynamic experimental system with 3 time-invariant control variables and 
1 response.
Problem: design optimal experiment for a order 2 polynomial, with complete interaction
Solution: a full 2^3 factorial design (2 level)
"""


def simulate(ti_controls, tv_controls, model_parameters, sampling_times):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        model_parameters[3] * ti_controls[2] +
        # 2-interaction term
        model_parameters[4] * ti_controls[0] * ti_controls[1] +
        model_parameters[5] * ti_controls[0] * ti_controls[2] +
        model_parameters[6] * ti_controls[1] * ti_controls[2] +
        # 3-interaction term
        model_parameters[7] * ti_controls[0] * ti_controls[1] * ti_controls[2]
    ])


designer_1 = Designer()
designer_1.simulate = simulate

tic_1, tic_2, tic_3 = np.mgrid[-1:1:11j, -1:1:11j, -1:1:11j]
tic_1 = tic_1.flatten()
tic_2 = tic_2.flatten()
tic_3 = tic_3.flatten()
designer_1.ti_controls_candidates = np.array([tic_1, tic_2, tic_3]).T

designer_1.model_parameters = np.ones(8)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

designer_1.design_experiment(designer_1.a_opt_criterion, write=False)
designer_1.print_optimal_candidates()
designer_1.plot_current_design()

fig1 = plt.figure()
axes1 = fig1.add_subplot(111, projection='3d')
axes1.scatter(designer_1.ti_controls_candidates[:, 0],
              designer_1.ti_controls_candidates[:, 1],
              designer_1.ti_controls_candidates[:, 2],
              s=designer_1.efforts * 1000)
axes1.grid(False)
axes1.set_title(r"Full $2^3$ Factorial Design")
axes1.set_xlabel("Control 1")
axes1.set_ylabel("Control 2")
axes1.set_zlabel("Experimental Effort")
axes1.set_xticks([-1, -.5, 0, .5, 1])
axes1.set_yticks([-1, -.5, 0, .5, 1])
axes1.set_zticks([-1, -.5, 0, .5, 1])
plt.show()
