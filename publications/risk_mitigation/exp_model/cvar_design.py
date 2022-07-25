from pydex.core.designer import Designer
from time import time
import multiprocessing as mp
import numpy as np


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] * np.exp(model_parameters[1] * ti_controls[0])
    ])

def my_func(x):
    designer = Designer()
    designer.simulate = simulate
    designer.ti_controls_candidates = [x[:1]]
    designer.model_parameters = [x[1:]]
    designer.initialize(verbose=0)
    sens = designer.eval_sensitivities()

    atom_fim = np.empty((designer.n_spt, designer.n_mp, designer.n_mp))
    for spt, s in enumerate(sens[0]):
        atom_fim[spt] = s.T @ designer.error_cov @ s

    return atom_fim

def main():
    reso = 21j
    tic = np.array([np.mgrid[0:0.5:reso]]).T

    np.random.seed(123)
    n_scr = 200
    param = np.random.uniform(
        low=[1, -10],
        high=[10, 0],
        size=(n_scr, 2),
    )
    parallel_input = np.empty((tic.shape[0] * param.shape[0], tic.shape[1] + param.shape[1]))
    parallel_input[:, 0] = np.tile(tic, n_scr).flatten(order="F")
    for s, par in enumerate(param):
        parallel_input[s*tic.shape[0]:(s+1)*tic.shape[0], 1:] = par

    print(
        f"Conducting sensitivity analysis for {tic.shape[0]:d} number of candidates and "
        f"{n_scr:d} number of parameter scenarios."
    )

    start = time()
    pool = mp.Pool(mp.cpu_count())
    result = np.asarray(pool.map(my_func, parallel_input))
    end = time()
    print(
        f"Parallel computation takes: {end - start:.2f} CPU seconds."
    )
    result = result.reshape((n_scr, tic.shape[0], result.shape[1], result.shape[3], result.shape[3]))

    outer_designer = Designer()
    outer_designer.simulate = simulate
    outer_designer.ti_controls_candidates = tic
    outer_designer.model_parameters = param
    outer_designer.start_logging()
    outer_designer.initialize(verbose=2)

    outer_designer.pb_atomic_fims = result
    outer_designer._model_parameters_changed = False
    outer_designer._candidates_changed = False

    outer_designer.solve_cvar_problem(
        outer_designer.cvar_d_opt_criterion,
        beta=0.75,
        plot=True,
        write=True,
        reso=10,
    )
    outer_designer.print_optimal_candidates()
    outer_designer.plot_pareto_frontier(write=True)
    outer_designer.stop_logging()
    outer_designer.show_plots()

if __name__ == '__main__':
    main()
