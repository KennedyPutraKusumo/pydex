from pydex.core.designer import Designer
import numpy as np


"""
Setting     : a non-dynamic experimental system with 2 time-invariant control variables 
              and 1 response.
Problem     : design optimal experiment for a order 1 polynomial.
Solution    : a 2^2 factorial design, criterion does not affect design.
"""

def simulate(ti_controls, model_parameters):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1]
    ])

designer = Designer()
designer.simulate = simulate
designer.model_parameters = [1, 2, 3]  # values won't affect design, but still needed
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=np.array([
        [-1, 1],
        [-1, 1],
    ]),
    levels=np.array([
        11,
        11,
    ])
)

designer.error_cov = np.diag([0.20])
designer._num_steps = 5
designer.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed
designer._norm_sens_by_params = True
designer.design_experiment(
    designer.d_opt_criterion,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True, write=False, markersize=3)
n_exps = [3]
seed = 123
for n_exp in n_exps:
    designer.apportion(n_exp=n_exp)
    mp_bounds = np.array([
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ])
    designer.insilico_bayesian_inference(
        n_walkers=32,
        n_steps=5000,
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
designer.show_plots()
