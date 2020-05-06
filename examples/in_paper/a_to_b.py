from pydex.core.designer import Designer
from matplotlib import pyplot as plt
import numpy as np

def simulate(ti_controls, sampling_times, model_parameters):
    return np.array([
        [
            np.exp(-model_parameters[0] * t),
            1 - np.exp(-model_parameters[0] * t)
        ]
        for t in sampling_times
    ])

designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.array([1])

designer.ti_controls_candidates = np.array([[1]])

n_spt = 100
spt_candidates = np.array([
    np.linspace(0, 10, n_spt)
])
designer.sampling_times_candidates = spt_candidates

designer.response_names = ["c_A", "c_B"]
designer.time_unit = "min"
designer.model_parameter_names = ["\\theta"]

designer.initialize()

criterion = designer.d_opt_criterion
designer.design_experiment(criterion, optimize_sampling_times=True)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_predictions(write=True)
designer.plot_optimal_sensitivities(write=True)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
axes.bar(spt_candidates[0], designer.efforts, width=10/n_spt)
axes.bar(spt_candidates[0], 1, width=10/n_spt, facecolor="none", edgecolor="black",
         alpha=0.05)
axes.set_xlabel("Sampling Time")
axes.set_ylabel("Effort")
fig.savefig("a_to_b_d_opt", dpi=720)
plt.show()
