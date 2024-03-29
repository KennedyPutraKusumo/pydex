from pydex.core.designer import Designer
import numpy as np
import sobol_seq


""" 
Setting     : a non-dynamic experimental system with 2 time-invariant control variables 
              and 1 response.
Problem     : design optimal experiment for a order 2 polynomial.
Solution    : 3^2 factorial design, varying efforts depending on chosen criterion:
              ~ D-optimal: well distributed.
              ~ A-optimal: slight central-focus.
              ~ E-optimal: strong central-focus.
"""

def simulate(ti_controls, model_parameters):
    return np.array([
        # Response 1 constant term
        model_parameters[0] +
        # Response 1 linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        # Response 1 linear-linear terms
        model_parameters[3] * ti_controls[0] * ti_controls[1] +
        # Response 1 squared terms
        model_parameters[4] * ti_controls[0] ** 2 +
        model_parameters[5] * ti_controls[1] ** 2
        ,
        # Response 2 constant term
        0.1 * (
            model_parameters[0] +
            # Response 2 linear term
            model_parameters[1] * np.exp(ti_controls[0]) +
            model_parameters[2] * np.exp(ti_controls[1]) +
            # Response 2 linear-linear terms
            model_parameters[3] * np.exp(ti_controls[0]) * np.exp(ti_controls[1]) +
            # Response 2 squared terms
            model_parameters[4] * np.exp(ti_controls[0]) ** 2 +
            model_parameters[5] * np.exp(ti_controls[1]) ** 2
        )
    ])


designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.ones(6)  # values won't affect design, but still needed
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

designer.start_logging()
designer.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed
designer.ti_controls_names = [r"$x_1$", r"$x_2$"]

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

designer.eval_sensitivities(method="central", num_steps=3)

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
designer.plot_optimal_controls(non_opt_candidates=True, title=True, write=False)

criterion = designer.a_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True, title=True, write=False)

criterion = designer.e_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True, title=True, write=False)

designer.stop_logging()
designer.show_plots()
