import numpy as np

from pydex.core.designer import Designer
from examples.non_cubic_spaces.experimental_spaces import triangle, heart, circle, folium

""" 
Setting: a non-dynamic experimental system with 2 time-invariant control variables and 
1 response.
Problem: design optimal experiment for a order 2 polynomial, with complete interaction
Solution: a full 3^2 factorial design (3 level)
"""


def simulate(ti_controls, model_parameters):
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

designer_1.model_parameters = np.ones(6)  # values won't affect design, but still needed

""" initializing initial grid """
reso = 41j
tic_1, tic_2 = np.mgrid[-1:1:reso, -1:1:reso]
tic_1 = tic_1.flatten()
tic_2 = tic_2.flatten()
tic = np.array([tic_1, tic_2]).T

package, optimizer = ("cvxpy", "MOSEK")
criterion = designer_1.a_opt_criterion

""" FOLIUM """
# filtering initial grid
folium_tic = folium(tic)
designer_1.ti_controls_candidates = folium_tic
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

# designing experiment
designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
                             write=False)

# visualize results
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_controls(non_opt_candidates=True)
designer_1.show_plots()

""" TRIANGLE """
# filtering initial grid
triangle_tic = triangle(tic)
designer_1.ti_controls_candidates = triangle_tic
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

# designing experiment
designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
                             write=False)

# visualize results
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_controls(non_opt_candidates=True)
designer_1.show_plots()

""" CIRCLE """
# filtering initial grid
circle_tic = circle(tic)
designer_1.ti_controls_candidates = circle_tic
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

# designing experiment
designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
                             write=False)

# visualize results
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_controls(non_opt_candidates=True)
designer_1.show_plots()

""" HEART """
# filtering initial grid
heart_tic = heart(tic)
designer_1.ti_controls_candidates = heart_tic
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

# designing experiment
designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
                             write=False)

# visualize results
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_controls(non_opt_candidates=True)
designer_1.show_plots()
