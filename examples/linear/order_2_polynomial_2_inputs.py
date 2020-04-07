from core.designer import Designer
from matplotlib import pyplot as plt
import numpy as np

""" 
Setting: a non-dynamic experimental system with 2 time-invariant control variables and 
1 response.
Problem: design optimal experiment for a order 2 polynomial, with complete interaction
Solution: a full 3^2 factorial design (3 level)
"""


def simulate(ti_controls, tv_controls, model_parameters, sampling_times):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        # interaction term
        model_parameters[3] * ti_controls[0] * ti_controls[1] +
        # squared terms
        model_parameters[4] * ti_controls[0] ** 2 +
        model_parameters[5] * ti_controls[1] ** 2
    ])


designer_1 = Designer()
designer_1.simulate = simulate

tic_1, tic_2 = np.mgrid[-1:1:11j, -1:1:11j]
tic_1 = tic_1.flatten()
tic_2 = tic_2.flatten()
designer_1.ti_controls_candidates = np.array([tic_1, tic_2]).T

designer_1.model_parameters = np.ones(6)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

package, optimizer = ("cvxpy", "MOSEK")
# package, optimizer = ("cvxpy", "SCS")
# package, optimizer = ("scipy", "SLSQP")
designer_1.design_experiment(designer_1.d_opt_criterion, package=package,
                             optimizer=optimizer,
                             write=False, fd_jac=False)
designer_1.print_optimal_candidates()
designer_1.plot_current_design()

fig1 = plt.figure()
axes1 = fig1.add_subplot(111)
axes1.scatter(designer_1.ti_controls_candidates[:, 0],
              designer_1.ti_controls_candidates[:, 1],
              s=np.round(designer_1.efforts * 1000, decimals=2))
plt.show()
