import numpy as np
from matplotlib import pyplot as plt

from core.designer import Designer

""" 
Setting: a non-dynamic experimental system with 3 time-invariant control variables and 1 response.
Problem: design optimal experiment for a order 2 polynomial, with complete interaction
Solution: D-optimal full 3^3 factorial design, A- and E-optimal Central Composite Design
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
        model_parameters[7] * ti_controls[0] * ti_controls[1] * ti_controls[2] +
        # squared terms
        model_parameters[8] * ti_controls[0] ** 2 +
        model_parameters[9] * ti_controls[1] ** 2 +
        model_parameters[10] * ti_controls[2] ** 2
    ])


designer_1 = Designer()
designer_1.simulate = simulate

reso = 9j
tic_1, tic_2, tic_3 = np.mgrid[-1:1:reso, -1:1:reso, -1:1:reso]
tic_1 = tic_1.flatten()
tic_2 = tic_2.flatten()
tic_3 = tic_3.flatten()
designer_1.ti_controls_candidates = np.array([tic_1, tic_2, tic_3]).T

designer_1.model_parameters = np.ones(11)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

""" cvxpy solvers """
package, optimizer = ("cvxpy", "MOSEK")
# package, optimizer = ("cvxpy", "SCS")
# package, optimizer = ("cvxpy", "CVXOPT")

""" scipy solvers, all supported, but many require unconstrained form """
# package, optimizer = ("scipy", "powell")
# package, optimizer = ("scipy", "cg")
# package, optimizer = ("scipy", "tnc")
# package, optimizer = ("scipy", "l-bfgs-b")
# package, optimizer = ("scipy", "bfgs")
# package, optimizer = ("scipy", "nelder-mead")
# package, optimizer = ("scipy", "SLSQP")  # supports constrained form

""" criterion choice """
criterion = designer_1.d_opt_criterion
# criterion = designer_1.a_opt_criterion
# criterion = designer_1.e_opt_criterion

""" designing experiment """
designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
                             write=False)
designer_1.print_optimal_candidates()
designer_1.plot_current_design()

fig1 = plt.figure()
axes1 = fig1.add_subplot(111, projection='3d')
axes1.scatter(
    designer_1.ti_controls_candidates[np.where(designer_1.efforts > 1e-4)][:, 0],
    designer_1.ti_controls_candidates[np.where(designer_1.efforts > 1e-4)][:, 1],
    designer_1.ti_controls_candidates[np.where(designer_1.efforts > 1e-4)][:, 2],
    s=designer_1.efforts[np.where(designer_1.efforts > 1e-4)] * 1000)
axes1.grid(False)
axes1.set_xlabel("Control 1")
axes1.set_ylabel("Control 2")
axes1.set_zlabel("Experimental Effort")
axes1.set_xticks([-1, -.5, 0, .5, 1])
axes1.set_yticks([-1, -.5, 0, .5, 1])
axes1.set_zticks([-1, -.5, 0, .5, 1])
plt.show()
