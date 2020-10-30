from pydex.core.designer import Designer
from real_model import candidates1, data1, candidates2, data2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


candidate = candidates1
data = data1
def simulate(ti_controls, model_parameters):
    return np.array([
        # response 1
        model_parameters[0] +
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        model_parameters[3] * ti_controls[0] ** 2 +
        model_parameters[4] * ti_controls[1] ** 2 +
        model_parameters[5] * ti_controls[0] * ti_controls[1],
        # response 2
        model_parameters[6] +
        model_parameters[7] * ti_controls[0] +
        model_parameters[8] * ti_controls[1] +
        model_parameters[9] * ti_controls[0] ** 2 +
        model_parameters[10] * ti_controls[1] ** 2 +
        model_parameters[11] * ti_controls[0] * ti_controls[1],
    ])

designer = Designer()
designer.simulate = simulate
designer.ti_controls_candidates = candidate
designer.data = data
designer.model_parameters = np.ones(12)
designer.initialize(verbose=2)
designer.estimate_parameters(
    init_guess=np.ones(12),
    bounds=[
        [-1e3, 1e3]
        for _ in range(12)
    ],
    write=False,
    update_parameters=True,
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
)
print("Parameter Estimates".center(100, "="))
print(designer.model_parameters)
print("Parameter Estimate Covariance".center(100, "="))
print(np.diag(designer.mp_covar))
designer.simulate_candidates()
print("Data".center(100, "="))
print(designer.data)
print("Predictions".center(100, "="))
print(designer.response)

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
