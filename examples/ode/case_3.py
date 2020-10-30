from pydex.core.designer import Designer
from case_3_model import simulate
import numpy as np


designer = Designer()
designer.simulate = simulate

# define designer state
if False:
    designer.ti_controls_candidates = designer.enumerate_candidates(
        bounds=[
            [1, 20],                # cA0
            [273.15, 323.15],       # T
            [1, 100],               # tau
        ],
        levels=[
            5,                     # cA0
            5,                     # T
            5,                     # tau
        ]
    )
    designer.sampling_times_candidates = [
        np.linspace(0, 1, 101)
        for _ in range(len(designer.ti_controls_candidates))
    ]
    designer.model_parameters = [
        5.4,
        5.0,
        6.2,
        0.5,
        1.4,
        2.5,
        7/3,
        3,
        5,
    ]
    designer.initialize(verbose=2)

# run sensitivity analysis, and save
if False:
    designer.save_state()
    designer.eval_sensitivities(save_sensitivities=True, num_steps=15,)
    designer.plot_sensitivities()
    designer.show_plots()

# load existing state, sensitivities, design experiment, and save
if False:
    designer.load_state("/case_3_result/date_2020-7-7/state_1_10x10x10.pkl")
    designer.load_sensitivity("/case_3_result/date_2020-7-7/sensitivity_1_10x10x10_15_steps.pkl")

    designer._regularize_fim = False
    designer.estimability_study_fim()

    criterion = designer.d_opt_criterion
    package, optimizer = "cvxpy", "MOSEK"
    # package, optimizer = "cvxpy", "SCS"
    # package, optimizer = "scipy", "l-bfgs-b"
    # package, optimizer = "scipy", "SLSQP"
    # package, optimizer = "scipy", "bfgs"

    if optimizer is "SCS":
        designer.design_experiment(
            designer.d_opt_criterion,
            write=True,
            optimize_sampling_times=False,
            package=package,
            optimizer=optimizer,
            max_iters=int(1e6),
            regularize_fim=False,
        )
    elif package is "SLSQP":
        designer.design_experiment(
            designer.d_opt_criterion,
            write=True,
            optimize_sampling_times=False,
            package=package,
            optimizer=optimizer,
            regularize_fim=False,
            opt_options={
                "disp"      : True,
                "maxiter"   : int(1e4),
            },
        )
    elif package is "l-bfgs-b":
        designer.design_experiment(
            designer.d_opt_criterion,
            write=True,
            optimize_sampling_times=False,
            package=package,
            optimizer=optimizer,
            regularize_fim=False,
            opt_options={
                "disp"      : True,
                "maxiter"   : int(1e4),
                "maxfun"    : int(1e10),
                "maxls"     : int(100),
            },
        )
    else:
        designer.design_experiment(
            designer.d_opt_criterion,
            write=True,
            optimize_sampling_times=False,
            package=package,
            optimizer=optimizer,
            regularize_fim=False,
        )
    designer.print_optimal_candidates()
    designer.plot_optimal_predictions()
    designer.plot_optimal_sensitivities()
    designer.show_plots()

# load optimal experimental design, print, and plot it
if True:
    designer.load_state("/case_3_result/date_2020-7-7/state_1_5x5x5.pkl")
    designer.initialize(verbose=2)
    designer.load_oed_result("/case_3_result/date_2020-7-7/d_opt_oed_result_1_5x5x5_15_steps.pkl")
    designer.load_sensitivity("/case_3_result/date_2020-7-7/sensitivity_1_5x5x5_15_steps.pkl")
    designer.print_optimal_candidates()
    designer.plot_optimal_sensitivities()
    designer.plot_optimal_predictions()
    designer.show_plots()
