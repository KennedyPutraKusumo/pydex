from pydex.core.designer import Designer
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
designer.start_logging()
designer.design_experiment(
    designer.d_opt_criterion,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()

n_exps = [80]
seed = 123
for n_exp in n_exps:
    designer.apportion(n_exp=n_exp)
    mp_bounds = np.array([
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
        ])
    designer.insilico_bayesian_inference(
        n_walkers=32,
        n_steps=10000,
        burn_in=100,
        bounds=mp_bounds,
        verbose=True,
        seed=seed,
    )
    fig = designer.plot_bayesian_inference_samples(
        contours=True,
        density=False,
        bounds=mp_bounds,
        title=f"{n_exp} Experiments, Seed: {seed}",
        plot_fim_confidence=True,
    )
    fig.savefig(f"bayesian_pe_{n_exp}_exp_{designer.error_cov[0][0]}_error.png")
    fig.tight_layout()
designer.stop_logging()
designer.show_plots()
