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

np.random.seed(1234)  # set seed for reproducibility
n_scr = 100
designer.model_parameters = np.random.normal(loc=1, scale=0.25, size=(n_scr, 1))

designer.ti_controls_candidates = np.array([[1]])

n_spt = 100
spt_candidates = np.array([
    np.linspace(0, 10, n_spt)
])
designer.sampling_times_candidates = spt_candidates

designer.initialize(verbose=1)

# sb_type = 0
sb_type = 1

criterion = designer.d_opt_criterion
# criterion = designer.a_opt_criterion
# criterion = designer.e_opt_criterion
designer.design_experiment(
    criterion,
    optimize_sampling_times=True,
    semi_bayes_type=sb_type,
    trim_fim=True,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_predictions(write=True)
designer.plot_optimal_sensitivities(absolute=False, write=True)

fig, axes = plt.subplots(1, 1)
axes.bar(spt_candidates[0], designer.efforts, width=10/n_spt)
axes.bar(spt_candidates[0], 1, width=10/n_spt, facecolor="none", edgecolor="black",
         alpha=0.1)
axes.set_xlabel("Sampling Time")
axes.set_ylabel("Effort")
fig.savefig("a_to_b_efforts", dpi=720)
plt.show()
