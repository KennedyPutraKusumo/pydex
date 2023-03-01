if __name__ == '__main__':
    import numpy as np
    from pydex.core.designer import Designer

    def dummy_sim(sampling_times, model_parameters):
        return np.array([
            0 for _ in sampling_times
        ])

    # read atomics from excel
    import pandas as pd
    data = pd.read_excel(
        "sensitivities_test.xlsx",
        sheet_name="Sheet2",
        header=6,
    )
    sensitivities = data[["Sens1", "Sens2"]].to_numpy()

    designer = Designer()
    designer.simulate = dummy_sim

    designer.sampling_times_candidates = np.array([
        data["Time [s]"].to_numpy()
    ])
    designer.model_parameters = np.ones(2)
    designer.initialize(verbose=2)

    # designer.load_sensitivity()
    designer.sensitivities = sensitivities[None, :, None, :]
    designer._model_parameters_changed = False
    designer._candidates_changed = False

    designer.design_experiment(
        designer.d_opt_criterion,
        optimize_sampling_times=True,
        n_spt=1,
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_sensitivities()
    designer.show_plots()
