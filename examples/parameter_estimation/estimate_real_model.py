from pydex.core.designer import Designer
from real_model import candidates1, data1
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


candidate = candidates1
data = data1
def simulate(ti_controls, model_parameters):
    x1, x2 = ti_controls
    return np.array([
        x1 * np.exp(-model_parameters[0] * x2),
        1 - x1 * np.exp(-model_parameters[0] * x2),
    ])

designer = Designer()
designer.simulate = simulate
designer.model_parameters = [5]
designer.ti_controls_candidates = candidate
designer.data = data
designer.initialize(verbose=2)
designer.estimate_parameters(
    bounds=[
        [-1e2, 1e2],
    ],
    write=False,
    update_parameters=True,
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
)
print("Parameter Estimates".center(100, "="))
print(designer.model_parameters)
print("Parameter Estimate Covariance".center(100, "="))
print(designer.mp_covar)

designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1],
        [-1, 1],
    ],
    levels=[
        11,
        11,
    ],
)
designer.initialize(verbose=2)
designer.simulate_candidates()
fig = plt.figure(figsize=(12, 4))

axes = fig.add_subplot(121, projection="3d")
axes.plot_trisurf(
    designer.ti_controls_candidates[:, 0],
    designer.ti_controls_candidates[:, 1],
    designer.response[:, 0],
)
axes.scatter(
    candidate[:, 0],
    candidate[:, 1],
    data[:, 0],
    c="red",
)
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_zlabel("z")
axes.grid(False)

axes2 = fig.add_subplot(122, projection="3d")
axes2.plot_trisurf(
    designer.ti_controls_candidates[:, 0],
    designer.ti_controls_candidates[:, 1],
    designer.response[:, 1],
)
axes2.scatter(
    candidate[:, 0],
    candidate[:, 1],
    data[:, 1],
    c="red",
)
axes2.set_xlabel("x")
axes2.set_ylabel("y")
axes2.set_zlabel("z")
axes2.grid(False)

fig.tight_layout()

plt.show()
