from pydex.core.designer import Designer
from Main_dummy import pydex_sim_dummy
import numpy as np

if __name__ == '__main__':
    designer_1 = Designer()
    designer_1.simulate = pydex_sim_dummy

    # designer_1.ti_controls_candidates = designer_1.enumerate_candidates(
    #     bounds=np.array([
    #         [20, 50],       # Temperature in Celsius
    #         [0.05, 0.95],   # organic modifier fraction phi vol/vol
    #     ]),
    #     levels=np.array([
    #         5,
    #         2,
    #     ])
    # )
    # designer_1.sampling_times_candidates = np.array([
    #     np.linspace(0, 50, 170) for _ in designer_1.ti_controls_candidates
    # ])

    designer_1.ti_controls_candidates = np.array([
    #   TEMP, PHI
        [25, 0.14],
        [20, 0.05],
        [35, 0.05],
        [50, 0.05],
        [20, 0.50],
        [35, 0.50],
        [50, 0.50],
        [20, 0.95],
        [35, 0.95],
        [50, 0.95],
    ])
    import pickle
    with open("case_1_result/variable_sampling_times.pkl", "rb") as file:
        designer_1.sampling_times_candidates = pickle.load(file)
    designer_1.model_parameters = np.array([
        40,     # a0
        12,
        8,
    ])
    designer_1.start_logging()
    designer_1.initialize(verbose=2)
    # designer_1.load_sensitivity(f"/case_1_result/date_2023-2-28/time_16-6-55/sensitivity_40_cand_1001_spt.pkl")
    designer_1.load_sensitivity(f"/case_1_result/analytical_sensitivity_case1.pkl")

    designer_1.sensitivities *= 10000
    if False:
        designer_1.sensitivities *= designer_1.model_parameters[None, None, None, :]
    designer_1.design_experiment(
        designer_1.d_opt_criterion,
        save_sensitivities=True,
        save_atomics=True,
        optimize_sampling_times=True,
    )
    designer_1.print_optimal_candidates()
    designer_1.plot_optimal_sensitivities(write=True)
    with open(f"case_1_result/response_matrix.pkl", "rb") as file:
        designer_1.response = pickle.load(file)
    designer_1.plot_optimal_predictions(write=True)
    designer_1.show_plots()
    designer_1.stop_logging()
