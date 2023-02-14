from pydex.core.designer import Designer
from case_2_model import simulate
import numpy as np


designer_1 = Designer()
designer_1.simulate = simulate

""" specifying nominal model parameter values """
pre_exp_constant = 0.1
activ_energy = 5000
theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
theta_1 = activ_energy / (8.314159 * 273.15)
theta_nom = np.array([theta_0, theta_1, 1, 0.5])  # value of theta_0, theta_1, alpha_a, nu
designer_1.model_parameters = theta_nom  # assigning it to the designer's theta

""" creating experimental candidates, here, it is generated as a grid """
tic = designer_1.enumerate_candidates(
    bounds=[
        [1, 5],             # initial C_A concentration
        [273.15, 323.15]    # reaction temperature
    ],
    levels=[
        5,                 # initial C_A concentration
        5,                 # reaction temperature
    ],
)
designer_1.ti_controls_candidates = tic

# defining sampling time candidates
spt_candidates = np.array([
    np.linspace(0, 200, 11)
    for _ in tic
])
designer_1.sampling_times_candidates = spt_candidates

if True:
    """
    =====================================================================================
    [Optional]
    1. Specify measurable states:
        A 1D array containing a subset of column numbers specifying the measurable states 
        from the response array returned by the simulate function. If un-specified, all 
        responses are assumed measurable.
    2. Name candidates, responses, and model parameters for plotting purposes.
        Adds titles to individual subplots whenever applicable.
    3. Save state.
        Saves the experimental candidates, nominal model parameter values to be loaded 
        for running related scripts in the future. State is saved into a pickle file.
    =====================================================================================
    """
    designer_1.measurable_responses = [0, 1]

    designer_1.candidate_names = np.array([
        f"Candidate {i+1}"
        for i, _ in enumerate(tic)
    ])
    designer_1.response_names = ["$c_A$", "$c_B$"]
    designer_1.model_parameter_names = [
        r"$\theta_0$",
        r"$\theta_1$",
        r"$\alpha$",
        r"$\nu$",
    ]

    # designer_1.save_state()

designer_1.error_cov = np.diag([0.1, 0.1])
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detail

""" Compute an optimal experiment design """
criterion = designer_1.d_opt_criterion
# criterion = designer_1.a_opt_criterion
package, optimizer = ("cvxpy", "MOSEK")
# package, optimizer = ("cvxpy", "SCS")
# package, optimizer = ("scipy", "SLSQP")
result = designer_1.design_experiment(
    criterion=criterion,
    optimize_sampling_times=True,  # ignoring sampling times
    n_spt=2,
    write=False,
    package=package,
    optimizer=optimizer,
)

designer_1.print_optimal_candidates()
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_sensitivities()
designer_1.show_plots()
designer_1.apportion(2)
# mp_bounds = np.array([
#     [-15, 0],
#     [0, 5],
#     [1, 2],
#     [0, 1],
# ])
# designer_1.insilico_bayesian_inference(
#     bounds=mp_bounds,
#     n_walkers=32,
#     n_steps=7000,
#     burn_in=150,
# )
# designer_1.plot_bayesian_inference_samples(
#     bounds=mp_bounds,
# )
#
# designer_1._optimization_package = "scipy"
# designer_1.compute_criterion_value(designer_1.a_opt_criterion)
# designer_1.compute_criterion_value(designer_1.e_opt_criterion)
# designer_1.compute_criterion_value(designer_1.dg_opt_criterion)
#
# redesign_result_2 = designer_1.design_experiment(
#     criterion=criterion,
#     optimize_sampling_times=True,   # sampling times as experimental variables
#     write=False,
#     package=package,
#     optimizer=optimizer,
# )
# designer_1.print_optimal_candidates()
# designer_1.plot_optimal_efforts()
# designer_1.apportion(12)
# designer_1.plot_optimal_predictions()
# designer_1.plot_optimal_sensitivities()
# designer_1.compute_criterion_value(designer_1.a_opt_criterion)
# designer_1.compute_criterion_value(designer_1.e_opt_criterion)
#
# redesign_result_2 = designer_1.design_experiment(
#     criterion=criterion,
#     optimize_sampling_times=True,   # sampling times as experimental variables
#     n_spt=2,                        # two sampling times
#     write=False,
#     package=package,
#     optimizer=optimizer,
# )
# designer_1.print_optimal_candidates()
# designer_1.apportion(12)
# designer_1.plot_optimal_predictions()
# designer_1.plot_optimal_sensitivities()
# designer_1.compute_criterion_value(designer_1.a_opt_criterion)
# designer_1.compute_criterion_value(designer_1.e_opt_criterion)

designer_1.show_plots()
