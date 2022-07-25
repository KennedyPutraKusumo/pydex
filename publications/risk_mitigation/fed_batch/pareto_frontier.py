from matplotlib import pyplot as plt
import numpy as np


iteration_results = np.array([
    [-0.45123293740587855, -2.656891379639002],
    [-0.3687418422232913, -2.6752618667508465],
    [-0.28625073545040003, -2.741673360403314],
    [-0.20375973830816915, -2.8855264774330474],
    [-0.12126859541262902, -3.315356348919587],
])
nominal_result = np.array([
    -2.820912, -5.06897615011683,
])
beta = 0.75

figsize = (4.5, 3.5)
fig = plt.figure(figsize=figsize)
axes = fig.add_subplot(111)
axes.plot(
    iteration_results[:, 0],
    iteration_results[:, 1],
    marker="o",
    label="Pareto-efficient Designs"
)
axes.set_xlabel("D-optimal Mean")
axes.set_ylabel(r"D-optimal CVaR ${}_{0.75}}$")
# axes.scatter(
#     nominal_result[0],
#     nominal_result[1],
#     label="Local Design",
#     marker="x",
#     c="tab:red",
# )
axes.legend()
fig.tight_layout()
plt.savefig("pareto.png", dpi=360)

sens_time = 3.64
opt_times = np.array([
    [3.64,  841.83,     1233.06],
    [0.00,  990.95,     1104.17],
    [0.00,  942.16,        0.00],
    [0.00,  948.12,        0.00],
    [0.00,  858.11,        0.00],
])


iteration_labels = [
    "Iter 1",
    "Iter 2",
    "Iter 3",
    "Iter 4",
    "Iter 5",
]
x = 3.0 * np.arange(len(iteration_labels))
width = 0.70
fig2 = plt.figure(figsize=(5, 5))
axes2 = fig2.add_subplot(111)
sens_bars = axes2.bar(
    x - width,
    opt_times[:, 0],
    width,
    label="Sensitivity",
)
opt_bars = axes2.bar(
    x,
    opt_times[:, 1],
    width,
    label="Optimization",
)
extra_bar = axes2.bar(
    x + width,
    opt_times[:, 2],
    width,
    label="Extra",
)
axes2.set_ylabel("CPU Time (s)")
axes2.set_xticks(x)
axes2.set_xticklabels(iteration_labels)
axes2.legend()
fig2.tight_layout()

plt.show()
