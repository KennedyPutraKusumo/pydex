from pydex.core.designer import Designer
from model import simulate_one_tic, simulate_tvc_feed
from model_order import simulate_order
import numpy as np


designer_1 = Designer()
calculate_local_average = False
if calculate_local_average:
    designer_local = Designer()

isothermal = False
tvc_feed = True
fix_switch = True
rxn_order = True
designer_1._save_txt = False
designer_1._save_txt_nc = 27
designer_1._save_txt_fmt = '% 7.2e'
designer_1._store_responses_rtol = 1e-12
designer_1._store_responses_atol = 0
designer_1.use_finite_difference = True
designer_1._num_steps = 5

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
                5,
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

designer_1.ti_controls_candidates = tic
designer_1.tv_controls_candidates = tvc

designer_1.sampling_times_candidates = np.array([
    np.linspace(0, 1, 11) for _ in range(tic.shape[0])
])

nominal_mp = [3.5e11 / 3600, 82500]
p_sdev = np.diag(nominal_mp * np.array([0.30, 0.05]))

np.random.seed(1)
n_scr = 100
model_parameters = np.random.multivariate_normal(nominal_mp, p_sdev, n_scr)
if rxn_order:
    rngorder = np.random.default_rng(seed=123)
    model_parameters = np.append(model_parameters, rngorder.choice([1, 2], (n_scr, 2), p=[0.75, 0.25]), axis=1)
designer_1.model_parameters = model_parameters
designer_1.model_parameters = model_parameters

designer_1.error_cov = np.diag([10, 1])

design_experiment = True
load_oed_result = False
if load_oed_result:
    design_experiment = False
    if tvc_feed:
        if fix_switch:
            designer_1.load_oed_result("/pb_oed_result/date_2022-5-28/run_1/run_1_d_opt_criterion_oed_result.pkl")
if calculate_local_average:
    designer_local.load_oed_result("/local_oed_result/date_2022-4-4/run_1/run_1_d_opt_criterion_oed_result.pkl")

designer_1.start_logging()
designer_1.initialize(verbose=2)
designer_1._norm_sens_by_params = True

save_sens = True
save_atoms = True
load_atomics = False
if load_atomics:
    save_sens = False
    save_atoms = False
    if tvc_feed:
        if fix_switch:
            if rxn_order:
                designer_1.load_atomics("/pb_oed_result/date_2022-5-28/run_1/run_1_atomics_1280_can_100_scr.pkl")

    if rxn_order:
        designer_1.pb_atomic_fims = designer_1.pb_atomic_fims.reshape(n_scr, 1280 * 11, 4, 4)

if design_experiment:
    designer_1.design_experiment(
        designer_1.d_opt_criterion,
        save_sensitivities=save_sens,
        optimize_sampling_times=False,
        save_atomics=save_atoms,
        pseudo_bayesian_type=1,
        write=True,
    )

if design_experiment or load_oed_result:
    designer_1.print_optimal_candidates(tol=1e-4)

if calculate_local_average:
    designer_1._optimization_package = "scipy"
    designer_1._pseudo_bayesian_type = 1
    average_value = -designer_1._pb_d_opt_criterion(designer_local.efforts)
    print("Average value of unrestricted local design ", average_value)
designer_1.stop_logging()
