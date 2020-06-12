from pydex.core.designer import Designer
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np


def simulate(ti_controls, tv_controls, sampling_times, model_parameters):
    y = odeint(
        dydt,
        [1, 0.01],
        sampling_times,
        args=(model_parameters, tv_controls)
    )
    return y

def dydt(y, t, k, u):
    u1_dict, u2_dict = u
    swt_ts_u1 = np.asarray(list(u1_dict.keys()))
    swt_ts_u2 = np.asarray(list(u2_dict.keys()))
    current_period_u1 = np.where(t >= swt_ts_u1, swt_ts_u1, np.inf).min()
    current_period_u2 = np.where(t >= swt_ts_u2, swt_ts_u2, np.inf).min()
    u1 = u1_dict[current_period_u1]
    u2 = u2_dict[current_period_u2]
    k1, k2, k3, k4 = k
    y1, y2 = y
    rm = k1 * y2 / (k2 + y2)
    yp = [
        y1 * (rm - k4 - u1),
        - rm * y1 / k3 + u1 * (u2 - y2)
    ]
    return yp

# def dydt(y, t, k, u):
#     u1_dict, u2_dict = u
#     swt_ts_u1 = np.asarray(list(u1_dict.keys()))
#     swt_ts_u2 = np.asarray(list(u2_dict.keys()))
#     current_period_u1 = np.where(t >= swt_ts_u1, swt_ts_u1, np.inf).min()
#     current_period_u2 = np.where(t >= swt_ts_u2, swt_ts_u2, np.inf).min()
#     u1 = u1_dict[current_period_u1]
#     u2 = u2_dict[current_period_u2]
#     k1, k2, k3, k4 = k
#     y1, y2 = y
#     rc = k1 * y2 / (k2 * y1 + y2)
#     yp = [
#         y1 * (rc - k4 - u1),
#         - rc * y1 / k3 + u1 * (u2 - y2)
#     ]
#     return yp

t = np.arange(0, 72, 0.75)
y = simulate([], [{0: 0.2}, {0: 35}], t, [0.31, 0.18, 0.55, 0.03])  # monod
# y = simulate([], [{0: 0.2}, {0: 35}], t, [0.30, 0.03, 0.55, 0.03])  # contois

fig = plt.figure()
axes = fig.add_subplot(111)
axes.scatter(t, y[:, 0], label="biomass")
axes.scatter(t, y[:, 1], label="substrate")
axes.set_xlim([0, 80])
axes.set_ylim([0, 30])
axes.legend()
plt.show()

designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.array([0.31, 0.18, 0.55, 0.03])
tic, tvc = designer.enumerate_candidates(
    np.array([
        [0, 1],
        [5, 35],
    ]),
    np.array([
        3,
        3,
    ]),
    np.array([
        np.linspace(0, 72, 5),
        np.linspace(0, 72, 1),
    ])
)
designer.ti_controls_candidates = tic
designer.tv_controls_candidates = tvc
designer.sampling_times_candidates = np.array([
    np.arange(0, 72, 0.75) for _ in range(tvc.shape[0])
])
designer.initialize(verbose=2)

criterion = designer.d_opt_criterion

designer.design_experiment(
    criterion,
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_predictions()
designer.plot_optimal_sensitivities()
designer.show_plots()
