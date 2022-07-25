from pydex.core.designer import Designer
from jacket_fed_batch import simulate2
import numpy as np
import pickle


designer = Designer()
designer.sens_report_freq = 1084
designer._save_txt = True
designer._save_txt_nc = 9
designer._save_txt_fmt = '% 7.2e'
designer._store_responses_rtol = 1e-12
designer._store_responses_atol = 0
designer.simulate = simulate2
with open("jfb_ds_0.95_case1_100_run/output.pkl", "rb") as file:
    ns_output = pickle.load(file)
alphas = [0.50, 0.80, 0.95]
samples = ns_output["solution"]["probabilistic_phase"]["samples"]
group1_samples = np.empty((0, 3))
group2_samples = np.empty((0, 3))
group3_samples = np.empty((0, 3))
group4_samples = np.empty((0, 3))
for coord, phi in zip(samples["coordinates"], samples["phi"]):
    if phi < alphas[0]:
        group1_samples = np.append(group1_samples, [coord], axis=0)
    elif phi < alphas[1]:
        group2_samples = np.append(group2_samples, [coord], axis=0)
    elif phi < alphas[2]:
        group3_samples = np.append(group3_samples, [coord], axis=0)
    else:
        group4_samples = np.append(group4_samples, [coord], axis=0)

n_cand = 424
sample_candidates = False
print_candidates = True
save_candidates = True
if sample_candidates:
    rng = np.random.default_rng(seed=123)
    candidates = rng.permutation(len(group4_samples))[:n_cand]
    tic = group4_samples[candidates]
    print("Number of randomly selected candidates: ", n_cand)
    if print_candidates:
        for rows in range(n_cand//10+1):
            for col in range(10):
                i_cand = rows*10+col
                if i_cand == n_cand:
                    break
                print(f"({i_cand+1},{candidates[i_cand]})", end=" ")
            print("")
    if save_candidates:
        fp = designer._generate_result_path("candidates", "pkl")
        with open(fp, "wb") as file:
            pickle.dump(candidates, file)
else:
    tic = group4_samples[:n_cand]
    print("Number of candidates: ", n_cand)

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

design_experiment = True
load_oed_result = False
if load_oed_result:
    design_experiment = False
    designer.load_oed_result("/jacket_fed_batch_oed_cc_result/date_2021-7-28/run_1/run_1_d_opt_criterion_oed_result.pkl")
    designer._opt_sampling_times = True
    designer._n_spt_spec = 1
    designer._cvar_problem = False
    designer._regularize_fim = False

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
        designer.load_sensitivity("/jacket_fed_batch_oed_cc_result/date_2021-4-14/run_1/run_1_sensitivity_257_cand_101_spt.pkl")
        save_sens = False
        save_atoms = False
    except FileNotFoundError:
        save_sens = True
        save_atoms = True
if False:
    try:
        designer.load_sensitivity(
            "/jacket_fed_batch_oed_cc_result/date_2021-4-16/run_1/run_1_sensitivity_210_cand_101_spt.pkl")
        save_sens = False
        save_atoms = False
    except FileNotFoundError:
        save_sens = True
        save_atoms = True
load_atomics = False
if load_atomics:
    try:
        designer.load_atomics(
            "/jacket_fed_batch_oed_cc_result/date_2021-7-28/run_1/run_1_atomics_424_cand_31_spt.pkl")
        designer.atomic_fims = designer.atomic_fims.reshape((424, 31, 5, 5))
        if sample_candidates:
            designer.atomic_fims = designer.atomic_fims[candidates, :31]
        else:
            designer.atomic_fims = designer.atomic_fims[:n_cand, :31]
        designer.atomic_fims = designer.atomic_fims.reshape((n_cand*31, 5, 5))
        designer.sensitivities = []
        save_sens = False
        save_atoms = False
    except FileNotFoundError:
        save_sens = True
        save_atoms = True
else:
    save_sens = True
    save_atoms = True
# designer._eps = 1e-5
designer.use_finite_difference = False
designer._num_steps = 15
if design_experiment:
    designer.design_experiment(
        criterion=designer.d_opt_criterion,
        save_sensitivities=save_sens,
        optimize_sampling_times=True,
        regularize_fim=False,
        save_atomics=save_atoms,
        write=True,
    )

if design_experiment or load_oed_result:
    designer.print_optimal_candidates(tol=1e-2)
    designer.plot_optimal_efforts(write=True, heatmap=True, figsize=(6.5, 6.5))

apportion = True
if apportion:
    apport_list = [5, 6, 7, 8, 9, 10, 20]
    for n_exp in apport_list:
        designer.apportion(n_exp)
designer.stop_logging()
designer.show_plots()
