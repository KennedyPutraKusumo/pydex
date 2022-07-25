from pydex.core.designer import Designer
from jacket_fed_batch import simulate2
import numpy as np


designer = Designer()
designer.sens_report_freq = 42
designer._save_txt = True
designer._save_txt_nc = 42
designer._save_txt_fmt = '% 7.2e'
designer._store_responses_rtol = 1e-12
designer._store_responses_atol = 0
designer.simulate = simulate2

tic = designer.enumerate_candidates(
    bounds=[
        [0.5, 2.5],                # switched q_in level
        [1.0, 20.0],                # duration in min
        [0.0, 5.00],                # q_w in L/min
    ],
    levels=[
        10,
        7,
        6,
    ],
)
designer.ti_controls_candidates = tic
designer.sampling_times_candidates = np.array([
    [t for t in np.linspace(0, 200, 201) if t >= 100 and t <= 130] for _ in tic
])

pre_exp_constant = 2.2e17
activ_energy = 1e5
theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
theta_1 = activ_energy / (8.314159 * 273.15)
mp = np.array([
    theta_0,
    theta_1,
    1.0,
    1.0,
    400,  # heat transfer coefficient in W.m-2.K-1
])
designer.model_parameters = mp

designer.error_cov = np.diag([
    0.05**2,    # concentration of A
    0.05**2,    # concentration of B
    0.1**2,     # reaction mixture Temperature
    0.1**2,     # Jacket Temperature
    0.01**2,    # Reaction Mixture Volume
])
designer.start_logging()
designer.initialize(verbose=2)

if False:
    try:
        designer.load_sensitivity("/jacket_fed_batch_oed_result/date_2021-3-25/run_1/run_1_sensitivity_243_cand_11_spt.pkl")
        save_sens = False
        save_atoms = False
    except FileNotFoundError:
        save_sens = True
        save_atoms = True
if False:
    try:
        designer.load_sensitivity("/jacket_fed_batch_oed_result/date_2021-4-9/run_1/run_1_sensitivity_27_cand_11_spt.pkl")
        save_sens = False
        save_atoms = False
    except FileNotFoundError:
        save_sens = True
        save_atoms = True
if False:
    try:
        designer.load_sensitivity("/jacket_fed_batch_oed_result/date_2021-4-13/run_4/run_4_sensitivity_27_cand_101_spt.pkl")
        save_sens = False
        save_atoms = False
    except FileNotFoundError:
        save_sens = True
        save_atoms = True
if True:
    save_sens = True
    save_atoms = True
# designer._eps = 1e-7
designer.use_finite_difference = False
designer._num_steps = 15
designer.design_experiment(
    criterion=designer.d_opt_criterion,
    save_sensitivities=save_sens,
    optimize_sampling_times=True,
    # regularize_fim=True,
    save_atomics=save_atoms,
)
designer.print_optimal_candidates(tol=1e-2)
designer.plot_optimal_efforts(write=True, heatmap=True, figsize=(6.5, 6.5))
designer.plot_optimal_predictions(write=True)
designer.plot_optimal_sensitivities(write=True)
apport_list = [5, 6, 7, 8, 9, 10, 20]
for n_exp in apport_list:
    designer.apportion(n_exp)
designer.stop_logging()
designer.show_plots()
