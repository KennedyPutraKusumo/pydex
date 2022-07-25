from pydex.core.designer import Designer
import numpy as np


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
    inner_designer = Designer()
    return_sensitivities = inner_designer.detect_sensitivity_analysis_function()
    res = np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        # 2-interaction term
        model_parameters[3] * ti_controls[0] * ti_controls[1] +
        # squared terms
        model_parameters[4] * ti_controls[0] ** 2 +
        model_parameters[5] * ti_controls[1] ** 2
    ])
    if return_sensitivities:
        sens = np.array([
            [
                [1, ti_controls[0], ti_controls[1], ti_controls[0] * ti_controls[1], ti_controls[0] ** 2, ti_controls[1] ** 2],
            ],
        ])
        return res, sens
    else:
        return res


designer = Designer()
designer.use_finite_difference = False
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
designer.error_cov = np.diag([4])
designer.initialize(verbose=3)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

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
designer._norm_sens_by_params = False
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls()
mp_bounds = np.array([
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
    [-10, 10],
])
n_exps = [6, 7, 8, 9, 10]
for n_exp in n_exps:
    designer.apportion(n_exp)
    designer.insilico_bayesian_inference(
        n_walkers=32,
        n_steps=5000,
        burn_in=100,
        bounds=mp_bounds,
        seed=123,
    )
    designer.plot_bayesian_inference_samples(
        bounds=mp_bounds,
        contours=True,
        density=False,
        plot_fim_confidence=True,
        write=True,
        reso=201j,
    )

criterion = designer.a_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls()

criterion = designer.e_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls()

designer.stop_logging()
designer.show_plots()
