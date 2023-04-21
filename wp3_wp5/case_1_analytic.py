from pydex.core.designer import Designer
from Main import pydex_sim
import numpy as np

if __name__ == '__main__':
    designer_1 = Designer()
    designer_1.simulate = pydex_sim
    designer_1.use_finite_difference = False

    designer_1.ti_controls_candidates = designer_1.enumerate_candidates(
        bounds=np.array([
            [20, 50],       # Temperature in Celsius
            [0.05, 0.95],   # organic modifier fraction phi vol/vol
        ]),
        levels=np.array([
            3,
            3,
        ])
    )
    designer_1.sampling_times_candidates = np.array([
        np.linspace(0, 50, 1001) for _ in designer_1.ti_controls_candidates
    ])
    designer_1.model_parameters = np.array([
        40,     # a0
        12,
        8,
    ])
    designer_1._norm_sens_by_params = True
    designer_1.error_cov = 1e-5 * np.eye(1)
    designer_1.start_logging()
    designer_1.initialize(verbose=2)
    designer_1.design_experiment(
        designer_1.d_opt_criterion,
        save_sensitivities=True,
        save_atomics=True,
        optimize_sampling_times=False,
    )
    designer_1.plot_optimal_sensitivities(write=True)
    designer_1.plot_optimal_predictions(write=True)
    designer_1.show_plots()
    designer_1.print_optimal_candidates()
    designer_1.stop_logging()
