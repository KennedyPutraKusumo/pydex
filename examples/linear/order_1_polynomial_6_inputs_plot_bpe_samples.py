from pydex.core.designer import Designer
from os import getcwd
import pickle
import numpy as np


"""
Setting     : a non-dynamic experimental system with 4 time-invariant control variables 
              and 1 response.
Problem     : design optimal experiment for a order 1 polynomial.
Solution    : a 2^6 factorial design.
"""

def simulate(ti_controls, model_parameters):
    inner_designer = Designer()
    return_sensitivities = inner_designer.detect_sensitivity_analysis_function()
    res = np.array([
        # constant
        model_parameters[0] +
        # linear
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        model_parameters[3] * ti_controls[2] +
        model_parameters[4] * ti_controls[3] +
        model_parameters[5] * ti_controls[4] +
        model_parameters[6] * ti_controls[5]
    ])
    if return_sensitivities:
        sens = np.array([
            [
                [1, ti_controls[0], ti_controls[1], ti_controls[2], ti_controls[3], ti_controls[4], ti_controls[5]]
            ],
        ])
        return res, sens
    else:
        return res

designer = Designer()
designer.use_finite_difference = False
designer.simulate = simulate
designer.model_parameters = np.ones(7)
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
    ],
    levels=[
        5,
        5,
        5,
        5,
        5,
        5,
    ],
)
designer.error_cov = np.diag([2])
designer.initialize(verbose=2)
designer.design_experiment(
    designer.d_opt_criterion,
)
n_exp = 70
designer.apportion(n_exp=n_exp)
with open(getcwd() + "/order_1_polynomial_6_inputs_result/date_2022-7-25/run_1/run_1_insilico_bayes_pe_samples_70_exp_32_walkers_10000_steps_100_burnin_123_seed.pkl", "rb") as file:
    bpe_samples = pickle.load(file)
designer.bayesian_pe_samples = bpe_samples
mp_bounds = np.array([
    [0, 2],
    [0, 2],
    [0, 2],
    [0, 2],
    [0, 2],
    [0, 2],
    [0, 2],
])
seed = 123
fig = designer.plot_bayesian_inference_samples(
    contours=True,
    density=False,
    bounds=mp_bounds,
    title=f"{n_exp} Experiments, Seed: {seed}",
    plot_fim_confidence=True,
)
fig.savefig(f"bayesian_pe_{n_exp}_exp_{designer.error_cov[0][0]}_error.png")
fig.tight_layout()
designer.show_plots()
