import numpy as np
import pickle
import os
from pydex.core.designer import Designer


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] * np.exp(model_parameters[1] * ti_controls[0])
    ])

designer = Designer()
designer.simulate = simulate

reso = 21j
tic = np.mgrid[0:1:reso]
designer.ti_controls_candidates = np.array([tic]).T

np.random.seed(123)
n_scr = 100
designer.model_parameters = np.random.multivariate_normal(
    mean=[1, -1],
    cov=np.array(
        [
            [0.2**2, 0.00],
            [0.00, 0.2**2],
        ]
    ),
    size=n_scr,
)
designer.initialize(verbose=2)

load = False
if not load:
    save_atomics = False
    designer.design_experiment(
        designer.e_opt_criterion,
        write=True,
        package="cvxpy",
        optimizer="MOSEK",
        pseudo_bayesian_type=1,
        save_atomics=save_atomics,
    )
else:
    designer.load_atomics("/pb_exp_model_result/date_2021-7-17/run_1/run_1_atomics_21_can_100_scr.pkl")
    designer.load_oed_result("/pb_exp_model_result/date_2021-7-17/run_1/run_1_d_opt_criterion_oed_result.pkl")
designer.print_optimal_candidates()
designer.plot_optimal_controls()
n_exps = [5, 6, 7, 8, 9, 10]
for n_exp in n_exps:
    designer.apportion(n_exp)

# designer.design_experiment(
#     designer.a_opt_criterion,
#     write=False,
#     package="cvxpy",
#     optimizer="MOSEK",
#     pseudo_bayesian_type=1,
# )
# designer.print_optimal_candidates()
# designer.plot_optimal_controls()
# for n_exp in n_exps:
#     designer.apportion(n_exp)

# designer.design_experiment(
#     designer.e_opt_criterion,
#     write=False,
#     package="cvxpy",
#     optimizer="MOSEK",
#     pseudo_bayesian_type=1,
# )
# designer.print_optimal_candidates()
# designer.plot_optimal_controls()
# for n_exp in n_exps:
#     designer.apportion(n_exp)

designer.show_plots()
