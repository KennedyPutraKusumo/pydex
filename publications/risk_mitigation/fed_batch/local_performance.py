from pydex.core.designer import Designer
from model import simulate
import numpy as np
import pickle


designer_1 = Designer()
designer_1.simulate = simulate

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
designer_1.model_parameters = theta_s

""" enumerating candidates """
designer_1.tv_controls_candidates = np.array([
    # Candidate 55
    [{0: 273.15}, {0: 0.1, 0.25: 0.00, 0.5: 0.00, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],
    # Candidate 61
    [{0: 273.15}, {0: 0.1, 0.25: 0.00, 0.5: 0.10, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],
    # Candidate 379
    [{0: 323.15}, {0: 0.1, 0.25: 0.00, 0.5: 0.00, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],
    # Candidate 58
    [{0: 273.15}, {0: 0.1, 0.25: 0.00, 0.5: 0.05, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],
    # Candidate 76
    [{0: 273.15}, {0: 0.1, 0.25: 0.10, 0.5: 0.05, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],

    # Candidate 382
    [{0: 323.15}, {0: 0.1, 0.25: 0.00, 0.5: 0.05, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],
    # Candidate 385
    [{0: 323.15}, {0: 0.1, 0.25: 0.00, 0.5: 0.10, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],
    # Candidate 400
    [{0: 323.15}, {0: 0.1, 0.25: 0.10, 0.5: 0.05, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],
    # Candidate 74
    [{0: 273.15}, {0: 0.1, 0.25: 0.10, 0.5: 0.00, 0.75: 0.05}, {0: 10.0}, {0: 0.0},],
    # Candidate 77
    [{0: 273.15}, {0: 0.1, 0.25: 0.10, 0.5: 0.05, 0.75: 0.05}, {0: 10.0}, {0: 0.0},],

    # Candidate 397
    [{0: 323.15}, {0: 0.1, 0.25: 0.10, 0.5: 0.00, 0.75: 0.00}, {0: 10.0}, {0: 0.0},],
    # Candidate 401
    [{0: 323.15}, {0: 0.1, 0.25: 0.10, 0.5: 0.05, 0.75: 0.05}, {0: 10.0}, {0: 0.0},],
    # Candidate 404
    [{0: 323.15}, {0: 0.1, 0.25: 0.10, 0.5: 0.10, 0.75: 0.05}, {0: 10.0}, {0: 0.0},],
    # Candidate 65
    [{0: 273.15}, {0: 0.1, 0.25: 0.05, 0.5: 0.00, 0.75: 0.05}, {0: 10.0}, {0: 0.0},],
    # Candidate 398
    [{0: 323.15}, {0: 0.1, 0.25: 0.10, 0.5: 0.00, 0.75: 0.05}, {0: 10.0}, {0: 0.0},],

    # Candidate 389
    [{0: 323.15}, {0: 0.1, 0.25: 0.05, 0.5: 0.00, 0.75: 0.05}, {0: 10.0}, {0: 0.0},],
])
designer_1.ti_controls_candidates = np.array([
    [5] for _ in designer_1.tv_controls_candidates
])
designer_1.sampling_times_candidates = np.array([
    [  0.,  20.,  40.,  60.,  80., 100., 120., 140., 160., 180., 200.]
    for _ in designer_1.tv_controls_candidates
])

designer_1.start_logging()
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detail

try:
    with open(f"local_performance_atom_fim_{designer_1.tv_controls_candidates.shape[0]}_n_c_{n_scr}_n_scr.pkl", "rb") as file:
        atom_fims = pickle.load(file)
    designer_1.pb_atomic_fims = atom_fims
    designer_1._model_parameters_changed = False
    designer_1._candidates_changed = False
    save_atom_fims = True
except FileNotFoundError:
    save_atom_fims = False

""" Computed Designs """
# Nominal design
spt = designer_1.sampling_times_candidates[0]
local_design_efforts = np.array([
        [0.3969 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.6031 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
        [0.0000 / len(spt) for _ in spt],
    ])

""" (optional) plotting attributes """
designer_1.response_names = ["c_A", "c_B"]
designer_1.model_parameter_names = ["\\theta_0", "\\theta_1", "\\alpha", "\\nu"]

pkg = "cvxpy"
opt = "MOSEK"

""" pb-D-optimal design """
criterion = designer_1.d_opt_criterion
result = designer_1.design_experiment(
    criterion=criterion,
    pseudo_bayesian_type=1,
    package=pkg,
    optimizer=opt,
    fix_effort=local_design_efforts,
    optimize_sampling_times=False,
)
designer_1.print_optimal_candidates()

""" CVaR D-optimal design """
criterion = designer_1.cvar_d_opt_criterion
result = designer_1.design_experiment(
    criterion=criterion,
    beta=0.80,
    package=pkg,
    optimizer=opt,
    fix_effort=local_design_efforts,
    optimize_sampling_times=False,
)
designer_1.print_optimal_candidates()
designer_1.plot_criterion_pdf()
designer_1.plot_criterion_cdf()

if save_atom_fims:
    with open(f"local_performance_atom_fim_{designer_1.ti_controls_candidates.shape[0]}_n_c_{n_scr}_n_scr.pkl", "wb") as file:
        pickle.dump(designer_1.pb_atomic_fims, file)

designer_1.stop_logging()
designer_1.show_plots()
