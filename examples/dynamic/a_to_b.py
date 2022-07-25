from pydex.core.designer import Designer
import numpy as np


def simulate(sampling_times, model_parameters):
    return np.array([
        [np.exp(-model_parameters[0] * t), 1 - np.exp(-model_parameters[0] * t)]
        for t in sampling_times
    ])

designer = Designer()
designer.simulate = simulate
designer.model_parameters = [1]
designer.sampling_times_candidates = np.array([
    np.linspace(0, 10, 11)
])
designer.initialize(verbose=2)
designer.response_names = ["c_A", "c_B"]
designer.model_parameter_names = ["\\theta"]

criterion = designer.d_opt_criterion
for n_spt in [1, 2]:
    designer.design_experiment(
        criterion,
        write=False,
        optimize_sampling_times=True,
        n_spt=n_spt,
        optimizer="SLSQP",
        package="scipy",
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_predictions()
    designer.plot_optimal_sensitivities()
    designer.apportion(4)
designer.show_plots()
