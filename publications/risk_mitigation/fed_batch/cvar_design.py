from pydex.core.designer import Designer
from time import time
from model import simulate
import multiprocessing as mp
import numpy as np
import pickle


def my_func(x):
    mini_designer = Designer()
    mini_designer.simulate = simulate
    mini_designer.ti_controls_candidates = np.array([x["tic"]])
    mini_designer.tv_controls_candidates = np.array([x["tvc"]])
    mini_designer.sampling_times_candidates = np.array([x["spt"]])
    mini_designer.model_parameters = x["mp"]
    mini_designer.initialize(verbose=0)
    sens = mini_designer.eval_sensitivities()

    atom_fim = np.empty((mini_designer.n_spt, mini_designer.n_mp, mini_designer.n_mp))
    for spt, s in enumerate(sens[0]):
        atom_fim[spt] = s.T @ mini_designer.error_fim @ s

    return atom_fim

def main():
    outer_designer = Designer()
    outer_designer.simulate = simulate
    tic, tvc = outer_designer.enumerate_candidates(
        bounds=[
            [5, 10],  # cA0
            [273.15, 323.15],  # temp
            [0, 1e-1],  # q_in in L/min
            [10, 20],  # ca_in in mol/L
            [0, 1],  # cb_in in mol/L
        ],
        levels=[
            1,
            3,
            3,
            1,
            1,
        ],
        switching_times=np.array([
            None,
            [0],
            [0, 0.25, 0.5, 0.75],
            [0],
            [0],
        ]),
    )
    outer_designer.ti_controls_candidates = tic
    outer_designer.tv_controls_candidates = tvc

    spt_candidates = np.array([np.linspace(0, 200, 11) for _ in tic])
    outer_designer.sampling_times_candidates = spt_candidates

    np.random.seed(123)
    n_scr = 200
    pre_exp_constant = np.random.uniform(0.1, 1.0, n_scr)
    activ_energy = np.random.uniform(1e3, 1e4, n_scr)
    theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
    theta_1 = activ_energy / (8.314159 * 273.15)
    theta_s = np.array([
        theta_0,
        theta_1,
        np.ones_like(pre_exp_constant),
        0.5 * np.ones_like(pre_exp_constant)
    ]).T
    outer_designer.model_parameters = theta_s

    parallel_input = []
    ticfull = np.tile(tic, (n_scr, 1))
    spt_candidates = np.tile(spt_candidates, (n_scr, 1))
    tvc = np.tile(tvc, (n_scr, 1))
    theta_s = np.tile(theta_s, (tic.shape[0], 1, 1)).reshape((tic.shape[0] * n_scr, 4), order="F")
    for ti, tv, spt, p in zip(ticfull, tvc, spt_candidates, theta_s):
        parallel_input.append({
            "tic": ti,
            "tvc": tv,
            "spt": spt,
            "mp": p,
        })

    print(
        f"Conducting sensitivity analysis for {tic.shape[0]:d} number of candidates and "
        f"{n_scr:d} number of parameter scenarios."
    )

    outer_designer.initialize(verbose=2)

    start = time()
    pool = mp.Pool(mp.cpu_count())
    result = np.asarray(pool.map(my_func, parallel_input))
    end = time()
    print(
        f"Parallel computation takes: {end - start:.2f} CPU seconds."
    )

    result = result.reshape((n_scr, tic.shape[0], result.shape[1], result.shape[3], result.shape[3]))

    return result, outer_designer


if __name__ == '__main__':
    computed_pb_atoms, outer_designer = main()

    with open(f"case2_tvc_atom_fims_{computed_pb_atoms.shape[1]:d}_n_c_{computed_pb_atoms.shape[0]:d}_n_scr.pkl", "wb") as file:
        pickle.dump(computed_pb_atoms, file)

    outer_designer.start_logging()

    outer_designer.pb_atomic_fims = computed_pb_atoms
    outer_designer._model_parameters_changed = False
    outer_designer._candidates_changed = False

    outer_designer.solve_cvar_problem(
        outer_designer.cvar_d_opt_criterion,
        beta=0.80,
        plot=True,
        write=True,
        optimize_sampling_times=True,
    )
    outer_designer.plot_pareto_frontier()

    outer_designer.stop_logging()

    outer_designer.show_plots()
