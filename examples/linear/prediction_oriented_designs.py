import numpy as np
from matplotlib import pyplot as plt

from pydex.core.designer import Designer

""" 
Setting: a non-dynamic experimental system with 2 time-invariant control variables and 1 response.
Problem: design a prediction-oriented experiment for order 1 polynomial with interaction
Solution: a full 2^2 factorial design (2 level)
"""
def simulate(ti_controls, model_parameters):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        # interaction term
        model_parameters[3] * ti_controls[0] * ti_controls[1]
    ])


designer_1 = Designer()
designer_1.simulate = simulate

tic_1, tic_2 = np.mgrid[-1:1:11j, -1:1:11j]
tic_1 = tic_1.flatten(); tic_2 = tic_2.flatten()
designer_1.ti_controls_candidates = np.array([tic_1, tic_2]).T

designer_1.model_parameters = np.ones(4)  # values won't affect design, but still needed

designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

""" scipy solvers required for prediction-oriented """
# package, optimizer = ("scipy", "powell")
# package, optimizer = ("scipy", "cg")
# package, optimizer = ("scipy", "tnc")
# package, optimizer = ("scipy", "l-bfgs-b")
# package, optimizer = ("scipy", "bfgs")
# package, optimizer = ("scipy", "nelder-mead")
package, optimizer = ("scipy", "SLSQP")  # supports constrained form

""" prediction-oriented criteria choices """
# criterion = designer_1.dg_opt_criterion
# criterion = designer_1.di_opt_criterion
# criterion = designer_1.ag_opt_criterion
# criterion = designer_1.ai_opt_criterion
criterion = designer_1.eg_opt_criterion
# criterion = designer_1.ei_opt_criterion

""" designing experiment """
designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
                             write=False)
designer_1.print_optimal_candidates()
designer_1.plot_current_design()

fig1 = plt.figure()
axes1 = fig1.add_subplot(111)
axes1.scatter(designer_1.ti_controls_candidates[:, 0],
              designer_1.ti_controls_candidates[:, 1],
              s=np.round(designer_1.efforts * 1000, decimals=2))
plt.show()
