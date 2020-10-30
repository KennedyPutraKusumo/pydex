from pydex.core.designer import Designer
import numpy as np


""" 
Setting     : a non-dynamic experimental system with 4 time-invariant control variables 
              and 1 response.
Problem     : design optimal experiment for a order 2 polynomial.
Solution    : 3^4 factorial design, varying efforts depending on chosen criterion:
              ~ D-optimal: well distributed.
              ~ A-optimal: slight central-focus.
              ~ E-optimal: strong central-focus.
"""

def simulate(ti_controls, model_parameters):
    return np.array([
        # constant
        model_parameters[0] +
        # linear
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        model_parameters[3] * ti_controls[2] +
        model_parameters[4] * ti_controls[3] +
        # linear-linear interaction
        model_parameters[5] * ti_controls[0] * ti_controls[1] +
        model_parameters[6] * ti_controls[0] * ti_controls[2] +
        model_parameters[7] * ti_controls[0] * ti_controls[3] +
        model_parameters[8] * ti_controls[1] * ti_controls[2] +
        model_parameters[9] * ti_controls[1] * ti_controls[3] +
        model_parameters[10] * ti_controls[2] * ti_controls[3] +
        # quadratic
        model_parameters[11] * ti_controls[0] ** 2 +
        model_parameters[12] * ti_controls[1] ** 2 +
        model_parameters[13] * ti_controls[2] ** 2 +
        model_parameters[14] * ti_controls[3] ** 2
    ])


designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.ones(15)  # values won't affect design, but still needed
outer_axes_reso = 11
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
    ],
    levels=[
        5,
        5,
        outer_axes_reso,
        outer_axes_reso,
    ],
)
designer.initialize(verbose=2)

""" cvxpy optimizers """
package, optimizer = ("cvxpy", "MOSEK")
# package, optimizer = ("cvxpy", "SCS")
# package, optimizer = ("cvxpy", "CVXOPT")  # only for A-optimal

""" scipy optimizers, all supported, but many require unconstrained form """
# package, optimizer = ("scipy", "powell")
# package, optimizer = ("scipy", "cg")
# package, optimizer = ("scipy", "tnc")
# package, optimizer = ("scipy", "l-bfgs-b")
# package, optimizer = ("scipy", "bfgs")
# package, optimizer = ("scipy", "nelder-mead")
# package, optimizer = ("scipy", "SLSQP")  # supports constrained form

""" designing experiment """
criterion = designer.d_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(intervals=[outer_axes_reso, outer_axes_reso])
designer.show_plots()

criterion = designer.a_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(intervals=[outer_axes_reso, outer_axes_reso])
designer.show_plots()

criterion = designer.e_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(intervals=[outer_axes_reso, outer_axes_reso])
designer.show_plots()
