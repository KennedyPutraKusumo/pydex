from pydex.core.designer import Designer
from scipy.integrate import odeint
import numpy as np


def simulate(ti_controls, tv_controls, sampling_times, model_parameters):
    def dydt(y, t, u, v, p):
        ca, cb = y
        theta0, theta1, alpha, nu = p
        temp_dict = u[0]
        temp_swt_ts = np.asarray(list(temp_dict.keys()))
        temp_current_period = np.where(temp_swt_ts <= t, temp_swt_ts, np.inf).min()
        temp = temp_dict[temp_current_period]
        tau = v
        k = np.exp(theta0 + theta1 * (temp - 273.15) / temp)
        dcadt = tau * (-k * ca ** alpha)
        dcbdt = - nu * dcadt
        yp = [
            dcadt,
            dcbdt,
        ]
        return yp
    y = odeint(dydt, [ti_controls[0], 0], sampling_times, args=(tv_controls, ti_controls[1], model_parameters))
    return y

designer = Designer()
designer.simulate = simulate
tic, tvc = designer.enumerate_candidates(
    bounds=[
        [1, 5],
        [50, 200],
        [273.15, 323.15],
    ],
    levels=[
        5,
        5,
        5,
    ],
    switching_times=[
        None,
        None,
        np.linspace(0, 200, 3),
    ]
)
designer.ti_controls_candidates = tic
designer.tv_controls_candidates = tvc
designer.sampling_times_candidates = np.array([
    np.linspace(0, 200, 11)
    for _ in range(tic.shape[0])
])
designer.model_parameters = np.array([-4.5, 2.2, 1, 0.5])
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
