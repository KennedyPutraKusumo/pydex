from pydex.core.designer import Designer
from model import simulate_one_tic, simulate_tvc_feed, simulate_tvc
from model_order import simulate_order
import numpy as np


designer_1 = Designer()
isothermal = False
tvc_feed = True
fix_switch = True
rxn_order = True
designer_1._save_txt = False
if isothermal:
    designer_1._save_txt_nc = 27
else:
    designer_1._save_txt_nc = 36
designer_1._save_txt_fmt = '% 7.2e'
designer_1._store_responses_rtol = 1e-12
designer_1._store_responses_atol = 0
designer_1.use_finite_difference = False
designer_1._num_steps = 10

if tvc_feed:
    if fix_switch:
        bounds = [
                     [65 + 273.15, 75 + 273.15],  # rxn temperature (K)
                     [0.0 / 3600, 0.1 / 3600],  # feedrate of B (L/s)
                 ]
    else:
        bounds = [
                     [65 + 273.15, 75 + 273.15],  # rxn temperature (K)
                     [0.03 / 3600, 0.1 / 3600],  # feedrate of B (L/s)
                     [0.1, 0.5],
                     [0.03 / 3600, 0.1 / 3600],
                     [0.0, 0.4]
                 ]
else:
    bounds = [
                 [65 + 273.15, 75 + 273.15],  # rxn temperature (K)
                 [0.03 / 3600, 0.1 / 3600],  # feedrate of B (L/s)
                 [0.1, 0.9]
             ]

if tvc_feed:
    if rxn_order:
        designer_1.simulate = simulate_order
    else:
        designer_1.simulate = simulate_tvc_feed
    if fix_switch:
        tic, tvc = designer_1.enumerate_candidates(
            bounds=bounds,
            levels=[
                4,
                4,
            ],
            switching_times=[
                None,
                [0, 0.25, 0.50, 0.75],
            ]
        )
    else:
        tic, tvc = designer_1.enumerate_candidates(
            bounds=bounds,
            levels=[
                5,
                3,
                5,
                3,
                5,
            ],
            switching_times=[
                None,
                None,
                None,
                None,
                None,
            ]
        )
elif isothermal:
    if rxn_order:
        designer_1.simulate = simulate_order
    else:
        designer_1.simulate = simulate_one_tic
    tic, tvc = designer_1.enumerate_candidates(
        bounds=bounds,
        levels=[
            9,
            11,
            11,
        ],
        switching_times=[
            None,
            None,
            None
        ]
    )
else:
    designer_1.simulate = simulate_tvc
    tic, tvc = designer_1.enumerate_candidates(
        bounds=bounds,
        levels=[
            2,
            11,
            11,
        ],
        switching_times=[
            [0.0, 0.33, 0.67],
            None,
            None
        ]
    )

designer_1.ti_controls_candidates = tic
designer_1.tv_controls_candidates = tvc

designer_1.sampling_times_candidates = np.array([
    np.linspace(0, 1, 11) for _ in range(tic.shape[0])
])

if rxn_order:
    designer_1.model_parameters = [
            3.5e11 / 3600,      # frequency factor k (L/(mol.s))
            82500,              # activation energy (J/mol)
            1,
            1,
    ]
else:
    designer_1.model_parameters = [
            3.5e11 / 3600,      # frequency factor k (L/(mol.s))
            82500,              # activation energy (J/mol)
    ]

designer_1.error_cov = np.diag([10, 1])

design_experiment = True
load_oed_result = False
if load_oed_result:
    design_experiment = False
    if tvc_feed:
        designer_1.load_oed_result("/local_oed_result/date_2022-3-17/run_2/run_2_d_opt_criterion_oed_result.pkl")
    elif isothermal:
        designer_1.load_oed_result("/local_oed_result/date_2022-3-17/run_1/run_1_d_opt_criterion_oed_result.pkl")

designer_1.start_logging()
designer_1.initialize(verbose=2)
designer_1._norm_sens_by_params = True
# designer_1._step_nom = np.abs(designer_1.model_parameters * 0.30)
# print(f"Custom nominal step used for finite difference: {designer_1._step_nom}")
# designer_1._num_steps = 10
# designer_1._step_nom = None
if design_experiment:
    designer_1.design_experiment(
        designer_1.d_opt_criterion,
        optimize_sampling_times=False,
        write=True,
    )
designer_1.print_optimal_candidates(tol=1e-4)
designer_1.plot_optimal_efforts(write=True, heatmap=True)
apportion_design = True
if apportion_design:
    apportion = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for n_exp in apportion:
        designer_1.apportion(n_exp)
    designer_1.stop_logging()
