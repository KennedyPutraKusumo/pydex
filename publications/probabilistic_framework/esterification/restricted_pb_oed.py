from pydex.core.designer import Designer
from model import simulate_one_tic, simulate_tvc_feed, simulate_tvc
from model_order import simulate_order
import numpy as np
import pickle

isothermal = False
tvc_feed = True
fix_switch = True
rxn_order = True

designer = Designer()
calculate_local_average = False
if calculate_local_average:
    designer_1 = Designer()

if tvc_feed:
    if fix_switch:
        designer.simulate = simulate_order
        if rxn_order:
            with open("restricted_space_samples_DEUS/output.pkl", "rb") as file:
                ns_output = pickle.load(file)
    else:
        designer.simulate = simulate_tvc_feed
        with open("22-03-18_1000_lp_100_n_scr-tvc_feed/output.pkl", "rb") as file:
            ns_output = pickle.load(file)
elif isothermal:
    designer.simulate = simulate_one_tic
    with open("22-03-02_1000_lp_100_n_scr/output.pkl", "rb") as file:
        ns_output = pickle.load(file)
else:
    designer.simulate = simulate_tvc
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
                     0.50: inside_samples_coords[i, 3], 0.75: inside_samples_coords[i,4]}]
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
                     0.50: inside_samples_coords[i, 3], 0.75: inside_samples_coords[i,4]}]
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

designer.ti_controls_candidates = tic
if tvc_feed:
    if fix_switch:
        designer.tv_controls_candidates = tvc
    else:
        designer.tv_controls_candidates = np.empty((tic.shape[0], 0))
elif isothermal:
    designer.tv_controls_candidates = np.empty((tic.shape[0], 0))
else:
    designer.tv_controls_candidates = tvc
designer.sampling_times_candidates = np.array([
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
designer.model_parameters = model_parameters

designer.error_cov = np.diag([10, 1])

design_experiment = True
load_oed_result = False
if load_oed_result:
    design_experiment = False
    if tvc_feed:
        if fix_switch:
            designer.load_oed_result("/restricted_pb_oed_result/date_2022-4-2/run_1/run_1_d_opt_criterion_oed_result.pkl")
        else:
            designer.load_oed_result("/restricted_pb_oed_result/date_2022-3-18/run_1/run_1_d_opt_criterion_oed_result.pkl")
    elif isothermal:
        if rxn_order:
            designer.load_oed_result("/restricted_pb_oed_result/date_2022-3-21/run_1/run_1_d_opt_criterion_oed_result.pkl")
        else:
            designer.load_oed_result("/restricted_pb_oed_result/date_2022-3-12/run_1/run_1_d_opt_criterion_oed_result.pkl")
    else:
        designer.load_oed_result("/restricted_pb_oed_result/date_2022-3-12/run_2/run_2_d_opt_criterion_oed_result.pkl")
if calculate_local_average:
    designer_1.load_oed_result("/restricted_local_oed_result/date_2022-4-4/run_1/run_1_d_opt_criterion_oed_result.pkl")

designer.start_logging()
designer.initialize(verbose=2)

if False:
    save_sens = True
    save_atoms = True
load_atomics = True
if load_atomics:
    save_sens = False
    save_atoms = False
    if tvc_feed:
        if fix_switch:
            if rxn_order:
                designer.load_atomics("/restricted_average_results/restricted_pb_atomic_fims_1080_n_c_100_n_scr_fix_switch_rxn_order.pkl")
        else:
            designer.load_atomics("/restricted_pb_atomic_fims_1028_n_c_100_n_scr_tvc_feed.pkl")
    elif isothermal:
        if rxn_order:
            designer.load_atomics("/restricted_pb_atomic_fims_1004_n_c_100_n_scr_order.pkl")
        else:
            designer.load_atomics("/restricted_pb_atomic_fims_1004_n_c_100_n_scr.pkl")
    else:
        designer.load_atomics("/restricted_pb_atomic_fims_1017_n_c_100_n_scr_tvc.pkl")
    if sample_candidates:
        designer.pb_atomic_fims = designer.pb_atomic_fims[:, candidates, :11]
    else:
        designer.pb_atomic_fims = designer.pb_atomic_fims[:, :n_cand, :11]
    if rxn_order:
        designer.pb_atomic_fims = designer.pb_atomic_fims.reshape(n_scr, n_cand * 11, 4, 4)
    else:
        designer.pb_atomic_fims = designer.pb_atomic_fims.reshape(n_scr, n_cand*11, 2, 2)

if design_experiment:
    designer.design_experiment(
        criterion=designer.d_opt_criterion,
        save_sensitivities=save_sens,
        optimize_sampling_times=False,
        regularize_fim=False,
        save_atomics=save_atoms,
        write=True,
        pseudo_bayesian_type=1,
    )

if design_experiment or load_oed_result:
    designer.print_optimal_candidates(tol=1e-4)
    designer.plot_optimal_efforts(write=True, heatmap=True)

apportion_design = True
if apportion_design:
    apportion = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for n_exp in apportion:
        designer.apportion(n_exp)

if calculate_local_average:
    designer._optimization_package = "scipy"
    average_value = -designer._pb_d_opt_criterion(designer_1.efforts)
    print("Average value of restricted local design ", average_value)

designer.stop_logging()
