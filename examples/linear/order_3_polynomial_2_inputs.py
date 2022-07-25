from pydex.core.designer import Designer
import numpy as np


""" 
Setting     : a non-dynamic experimental system with 2 time-invariant control variables 
              and 1 response.
Problem     : design optimal experiment for a order 3 polynomial.
Solution    : non-standard design with 16 candidates, varies with criterion.
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
        model_parameters[5] * ti_controls[1] ** 2 +
        # cubic terms
        model_parameters[6] * ti_controls[0] ** 2 * ti_controls[1] +
        model_parameters[7] * ti_controls[1] ** 2 * ti_controls[0] +
        model_parameters[8] * ti_controls[0] ** 3 +
        model_parameters[9] * ti_controls[1] ** 3
    ])


designer_1 = Designer()
designer_1.simulate = simulate
designer_1.model_parameters = np.ones(10)  # values won't affect design, but still needed
designer_1.ti_controls_candidates = designer_1.enumerate_candidates(
    bounds=[
        [-1, 1],
        [-1, 1],
    ],
    levels=[
        21,
        21,
    ],
)
designer_1.error_cov = np.diag([1.0])
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

""" designing experiment """
criterion = designer_1.d_opt_criterion
designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
                             write=False)

designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_controls(non_opt_candidates=True)
mp_bounds = np.array([
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
])
n_exps = [48]
for n_exp in n_exps:
    designer_1.apportion(n_exp)
    designer_1.insilico_bayesian_inference(
        n_walkers=32,
        n_steps=10000,
        burn_in=100,
        bounds=mp_bounds,
        seed=123,
    )
    designer_1.plot_bayesian_inference_samples(
        bounds=mp_bounds,
        contours=True,
        density=False,
        plot_fim_confidence=True,
        write=True,
        reso=201j,
    )

# criterion = designer_1.a_opt_criterion
# designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
#                              write=False)
#
# designer_1.print_optimal_candidates()
# designer_1.plot_optimal_efforts()
# designer_1.plot_optimal_controls(non_opt_candidates=True)
#
# criterion = designer_1.e_opt_criterion
# designer_1.design_experiment(criterion=criterion, package=package, optimizer=optimizer,
#                              write=False)
#
# designer_1.print_optimal_candidates()
# designer_1.plot_optimal_efforts()
# designer_1.plot_optimal_controls(non_opt_candidates=True)
designer_1.show_plots()
