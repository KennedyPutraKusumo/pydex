from pydex.core.designer import Designer
from case_2_tvc_model import simulate
import numpy as np


designer_1 = Designer()
designer_1.simulate = simulate

""" specifying nominal model parameter values """
pre_exp_constant = 0.1
activ_energy = 5000
theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
theta_1 = activ_energy / (8.314159 * 273.15)
theta_nom = np.array([theta_0, theta_1, 1, 0.5])  # value of theta_0, theta_1, alpha_a, nu
designer_1.model_parameters = theta_nom  # assigning it to the designer's theta

""" enumerating candidates """
tic, tvc = designer_1.enumerate_candidates(
    bounds=[
        [5, 10],            # cA0
        [273.15, 323.15],   # temp
        [0, 1e-1],          # q_in in L/min
        [10, 20],           # ca_in in mol/L
        [0, 1],             # cb_in in mol/L
    ],
    levels=[
        1,
        5,
        3,
        1,
        1,
    ],
    switching_times=np.array([
        None,
        [0],
        [0, 0.25, 0.50, 0.75],
        [0],
        [0],
    ]),
)
designer_1.ti_controls_candidates = tic
designer_1.tv_controls_candidates = tvc

spt_candidates = np.array([np.linspace(0, 200, 11) for _ in range(tic.shape[0])])
designer_1.sampling_times_candidates = spt_candidates

designer_1.start_logging()
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detail

""" (optional) plotting attributes """
designer_1.response_names = ["c_A", "c_B"]
designer_1.model_parameter_names = ["\\theta_0", "\\theta_1", "\\alpha", "\\nu"]

""" D-optimal design """
criterion = designer_1.d_opt_criterion
result = designer_1.design_experiment(
    criterion=criterion,
    package="cvxpy",
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_predictions(colour_map="rainbow")
designer_1.plot_optimal_sensitivities(colour_map="rainbow")
designer_1.stop_logging()

designer_1.show_plots()
