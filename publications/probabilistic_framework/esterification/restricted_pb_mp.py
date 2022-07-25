from pydex.core.designer import Designer
from time import time
import sys
import pickle
import multiprocessing as mp
import numpy as np
from model import simulate_one_tic, simulate_tvc_feed, simulate_tvc
from model_order import simulate_order

isothermal = False
tvc_feed = True
fix_switch = True
rxn_order = True

if tvc_feed:
    if fix_switch:
        if rxn_order:
            with open("restricted_space_samples_DEUS/output.pkl", "rb") as file:
                ns_output = pickle.load(file)
    else:
        with open("22-03-18_1000_lp_100_n_scr-tvc_feed/output.pkl", "rb") as file:
            ns_output = pickle.load(file)
elif isothermal:
    with open("22-03-02_1000_lp_100_n_scr/output.pkl", "rb") as file:
        ns_output = pickle.load(file)
else:
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
print_candidates = False
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

def eval_sensitivities_func(x):
    mini_designer = Designer()
    mini_designer.use_finite_difference = False
    if tvc_feed:
        if fix_switch:
            mini_designer.simulate = simulate_order
        else:
            mini_designer.simulate = simulate_tvc_feed
    else:
        if rxn_order:
            mini_designer.simulate = simulate_order
        else:
            mini_designer.simulate = simulate_one_tic
    mini_designer.ti_controls_candidates = tic
    if fix_switch:
        mini_designer.tv_controls_candidates = tvc
    else:
        mini_designer.tv_controls_candidates = np.empty((tic.shape[0], 0))
    mini_designer.sampling_times_candidates = np.array([
        np.linspace(0, 1, 11) for _ in range(tic.shape[0])
    ])
    mini_designer.model_parameters = x
    mini_designer.error_cov = np.diag([10, 1])
    mini_designer.initialize(verbose=2)
    mini_designer._norm_sens_by_params = True
    sens = mini_designer.eval_sensitivities()

    atom_fim = np.empty((mini_designer.n_c, mini_designer.n_spt, mini_designer.n_mp, mini_designer.n_mp))
    for c, sens2 in enumerate(sens):
        for spt, s in enumerate(sens2):
            atom_fim[c, spt] = s.T @ mini_designer.error_fim @ s

    return atom_fim


def eval_sensitivities_tvc(x):
    mini_designer = Designer()
    mini_designer.use_finite_difference = False
    mini_designer.simulate = simulate_tvc
    mini_designer.ti_controls_candidates = tic
    mini_designer.tv_controls_candidates = tvc
    mini_designer.sampling_times_candidates = np.array([
        np.linspace(0, 1, 11) for _ in range(tic.shape[0])
    ])
    mini_designer.model_parameters = x
    mini_designer.error_cov = np.diag([10, 1])
    mini_designer.initialize(verbose=2)
    mini_designer._norm_sens_by_params = True
    sens = mini_designer.eval_sensitivities()

    atom_fim = np.empty((mini_designer.n_c, mini_designer.n_spt, mini_designer.n_mp, mini_designer.n_mp))
    for c, sens2 in enumerate(sens):
        for spt, s in enumerate(sens2):
            atom_fim[c, spt] = s.T @ mini_designer.error_fim @ s

    return atom_fim


def eval_scenarios():

    nominal_mp = [3.5e11 / 3600, 82500]
    p_sdev = np.diag(nominal_mp * np.array([0.30, 0.05]))

    np.random.seed(1)
    n_scr = 100
    model_parameters = np.random.multivariate_normal(nominal_mp, p_sdev, n_scr)
    if rxn_order:
        rngorder = np.random.default_rng(seed=123)
        model_parameters = np.append(model_parameters, rngorder.choice([1, 2], (n_scr, 2), p=[0.75, 0.25]), axis=1)

    print(
        f"Conducting sensitivity analysis for {tic.shape[0]:d} candidates and "
        f"{n_scr:d} parameter scenarios."
    )

    calculate_atomic_fim = True
    if calculate_atomic_fim:
        start = time()
        parallel_processing = True
        if tvc_feed or isothermal:
            eval_func = eval_sensitivities_func
        else:
            eval_func = eval_sensitivities_tvc
        if parallel_processing:
            pool = mp.Pool(int(sys.argv[2]))
            result = np.asarray(pool.map(eval_func, model_parameters))
        else:
            result = []
            for param in model_parameters:
                result.append(eval_func(param))
            result = np.asarray(result)
        end = time()
        if parallel_processing:
            print(f"Parallel computation takes: {end - start:.2f} CPU seconds.")
        else:
            print(f"Serial computation takes: {end - start:.2f} CPU seconds.")

        if tvc_feed:
            if fix_switch:
                if rxn_order:
                    with open(f"restricted_pb_atomic_fims_{result.shape[1]:d}_n_c_{result.shape[0]:d}_n_scr_fix_switch_rxn_order.pkl", "wb") as pklfile:
                        pickle.dump(result, pklfile)
            else:
                with open(f"restricted_pb_atomic_fims_{result.shape[1]:d}_n_c_{result.shape[0]:d}_n_scr_tvc_feed.pkl", "wb") as pklfile:
                    pickle.dump(result, pklfile)
        elif isothermal:
            if rxn_order:
                with open(f"restricted_pb_atomic_fims_{result.shape[1]:d}_n_c_{result.shape[0]:d}_n_scr_order.pkl", "wb") as pklfile:
                    pickle.dump(result, pklfile)
            else:
                with open(f"restricted_pb_atomic_fims_{result.shape[1]:d}_n_c_{result.shape[0]:d}_n_scr.pkl", "wb") as pklfile:
                    pickle.dump(result, pklfile)
        else:
            with open(f"restricted_pb_atomic_fims_{result.shape[1]:d}_n_c_{result.shape[0]:d}_n_scr_tvc.pkl", "wb") as pklfile:
                pickle.dump(result, pklfile)

    design_experiment = False
    if design_experiment:
        if calculate_atomic_fim:
            result = result.reshape((n_scr, tic.shape[0]*result.shape[2], result.shape[3], result.shape[3]))

        outer_designer = Designer()
        outer_designer.use_finite_difference = False
        if tvc_feed:
            if fix_switch:
                outer_designer.simulate = simulate_order
            else:
                outer_designer.simulate = simulate_tvc_feed
            outer_designer.ti_controls_candidates = tic
            if fix_switch:
                outer_designer.tv_controls_candidates = tvc
            else:
                outer_designer.tv_controls_candidates = np.empty((tic.shape[0], 0))
        elif isothermal:
            if rxn_order:
                outer_designer.simulate = simulate_order
            else:
                outer_designer.simulate = simulate_one_tic
            outer_designer.ti_controls_candidates = tic
            outer_designer.tv_controls_candidates = np.empty((tic.shape[0], 0))
        else:
            outer_designer.simulate = simulate_tvc
            outer_designer.ti_controls_candidates = tic
            outer_designer.tv_controls_candidates = tvc
        outer_designer.sampling_times_candidates = np.array([
            np.linspace(0, 1, 11) for _ in range(tic.shape[0])
        ])
        outer_designer.model_parameters = model_parameters
        outer_designer.error_cov = np.diag([10, 1])
        outer_designer.start_logging()
        outer_designer.initialize(verbose=2)

        if calculate_atomic_fim:
            outer_designer.pb_atomic_fims = result
            outer_designer._model_parameters_changed = False
            outer_designer._candidates_changed = False
        else:
            outer_designer._norm_sens_by_params = True

        outer_designer.design_experiment(
            criterion=outer_designer.d_opt_criterion,
            optimize_sampling_times=False,
            regularize_fim=False,
            write=True,
            pseudo_bayesian_type=1,
        )
        outer_designer.print_optimal_candidates(tol=1e-4)
        outer_designer.stop_logging()

if __name__ == '__main__':
    eval_scenarios()
