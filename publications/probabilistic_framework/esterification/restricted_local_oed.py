from pydex.core.designer import Designer
from model import simulate_one_tic, simulate_tvc_feed, simulate_tvc
from model_order import simulate_order
import numpy as np
import pickle

isothermal = False
tvc_feed = True
fix_switch = True
rxn_order = True

designer_1 = Designer()
designer_1._save_txt = False
designer_1._save_txt_nc = 9
designer_1._save_txt_fmt = '% 7.2e'
designer_1._store_responses_rtol = 1e-12
designer_1._store_responses_atol = 0
if tvc_feed:
    if fix_switch:
        designer_1.simulate = simulate_order
        if rxn_order:
            with open("restricted_space_samples_DEUS/output.pkl", "rb") as file:
                ns_output = pickle.load(file)
    else:
        designer_1.simulate = simulate_tvc_feed
        with open("22-03-18_1000_lp_100_n_scr-tvc_feed/output.pkl", "rb") as file:
            ns_output = pickle.load(file)
elif isothermal:
    designer_1.simulate = simulate_one_tic
    with open("22-03-02_1000_lp_100_n_scr/output.pkl", "rb") as file:
        ns_output = pickle.load(file)
else:
    designer_1.simulate = simulate_tvc
    with open("22-03-06_1000_lp_100_n_scr_tvc/output.pkl", "rb") as file:
        ns_output = pickle.load(file)

target_reliability = 0.95
samples = ns_output["solution"]["probabilistic_phase"]["samples"]
inside_samples_coords = []
outside_samples_coords = []
for i, phi in enumerate(samples["phi"]):
    if phi >= target_reliability:
        inside_samples_coords.append(samples["coordinates"][i])
    else:
        outside_samples_coords.append(samples["coordinates"][i])
inside_samples_coords = np.array(inside_samples_coords)
outside_samples_coords = np.array(outside_samples_coords)

n_cand = inside_samples_coords.shape[0]
sample_candidates = False
print_candidates = True
if sample_candidates:
    rng = np.random.default_rng(seed=123)
    candidates = rng.permutation(len(inside_samples_coords))[:n_cand]
    if tvc_feed:
        if fix_switch:
            tic = inside_samples_coords[candidates, :1]
            tvc = [[{0.0: inside_samples_coords[i, 1], 0.25: inside_samples_coords[i, 2],
                     0.50: inside_samples_coords[i, 3], 0.75: inside_samples_coords[i, 4]}]
                   for i in candidates]
        else:
            tic = inside_samples_coords[candidates]
    elif isothermal:
        tic = inside_samples_coords[candidates]
    else:
        tic = inside_samples_coords[candidates, 3:5]
        tvc = [[{0.0: inside_samples_coords[i, 0], 0.33: inside_samples_coords[i, 1], 0.67: inside_samples_coords[i, 2]}]
               for i in candidates]
    print("Number of randomly selected candidates: ", n_cand)
    if print_candidates:
        for rows in range(n_cand//10+1):
            for col in range(10):
                i_cand = rows*10+col
                if i_cand == n_cand:
                    break
                print(f"({i_cand+1},{candidates[i_cand]})", end=" ")
            print("")
else:
    if tvc_feed:
        if fix_switch:
            tic = inside_samples_coords[:n_cand, :1]
            tvc = [[{0.0: inside_samples_coords[i, 1], 0.25: inside_samples_coords[i, 2],
                     0.50: inside_samples_coords[i, 3], 0.75: inside_samples_coords[i, 4]}]
                   for i in range(n_cand)]
        else:
            tic = inside_samples_coords[:n_cand]
    elif isothermal:
        tic = inside_samples_coords[:n_cand]
    else:
        tic = inside_samples_coords[:n_cand, 3:5]
        tvc = [[{0.0: inside_samples_coords[i, 0], 0.33: inside_samples_coords[i, 1], 0.67: inside_samples_coords[i, 2]}]
               for i in range(n_cand)]
    print("Number of candidates: ", n_cand)

designer_1.ti_controls_candidates = tic
if tvc_feed:
    if fix_switch:
        designer_1.tv_controls_candidates = tvc
    else:
        designer_1.tv_controls_candidates = np.empty((tic.shape[0], 0))
elif isothermal:
    designer_1.tv_controls_candidates = np.empty((tic.shape[0], 0))
else:
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
        if fix_switch:
            designer_1.load_oed_result("/restricted_local_oed_result/date_2022-4-4/run_1/run_1_d_opt_criterion_oed_result.pkl")
    elif isothermal:
        designer_1.load_oed_result("/restricted_local_oed_result/date_2022-3-12/run_1/run_1_d_opt_criterion_oed_result.pkl")

designer_1.start_logging()
designer_1.initialize(verbose=2)
designer_1._norm_sens_by_params = True

load_atomics = False
if load_atomics:
    try:
        if tvc_feed:
            if fix_switch:
                designer_1.load_atomics("/restricted_local_oed_result/date_2022-4-4/run_1/run_1_atomics_1080_cand_11_spt.pkl")
        elif isothermal:
            designer_1.load_atomics("/restricted_local_oed_result/date_2022-3-2/run_1/run_1_atomics_1004_cand_11_spt.pkl")
        if sample_candidates:
            designer_1.atomic_fims = designer_1.atomic_fims[candidates, :11]
        else:
            designer_1.atomic_fims = designer_1.atomic_fims[:n_cand, :11]
        if rxn_order:
            designer_1.atomic_fims = designer_1.atomic_fims.reshape((n_cand * 11, 4, 4))
        else:
            designer_1.atomic_fims = designer_1.atomic_fims.reshape((n_cand * 11, 2, 2))
        designer_1.sensitivities = []
        save_sens = False
        save_atoms = False
    except FileNotFoundError:
        save_sens = True
        save_atoms = True
else:
    save_sens = True
    save_atoms = True

designer_1.use_finite_difference = False

if design_experiment:
    designer_1.design_experiment(
        criterion=designer_1.d_opt_criterion,
        save_sensitivities=save_sens,
        optimize_sampling_times=False,
        regularize_fim=False,
        save_atomics=save_atoms,
        write=True,
    )

if design_experiment or load_oed_result:
    designer_1.print_optimal_candidates(tol=1e-4)
    designer_1.plot_optimal_efforts(write=True, heatmap=True)

apportion_design = True
if apportion_design:
    apportion = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for n_exp in apportion:
        designer_1.apportion(n_exp)

designer_1.stop_logging()
