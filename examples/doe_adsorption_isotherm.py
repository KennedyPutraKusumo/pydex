# https://github.com/KennedyPutraKusumo/pydex/blob/master/examples/ode/case_1.py
# https://github.com/KennedyPutraKusumo/pydex/blob/master/examples/ode/case_1_model.py

from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt
from pydex.core.designer import Designer


def scipy_simulate(ti_controls, sampling_times, model_parameters):
    def scipy_model(t, ca, theta):
        dca_dt = -theta[0] * ca
        return dca_dt

    sol = solve_ivp(scipy_model, [0, np.max(sampling_times)], [ti_controls[0]],
                    t_eval=sampling_times, args=(model_parameters,))

    return sol.y.T


scipy_simulate([1], [0, 2, 4, 6, 8], [2])

from pydex.core.designer import Designer

designer_1 = Designer()
designer_1.simulate = scipy_simulate

tic = designer_1.enumerate_candidates(
    bounds=[[0.1, 5]],
    levels=[5, ],
)

tic

designer_1.ti_controls_candidates = tic
designer_1.sampling_times_candidates = np.array([np.linspace(0, 50, 101) for _ in tic])
designer_1.model_parameters = np.array([0.25])

designer_1.initialize(verbose=2)

package, optimizer = ('scipy', 'SLSQP')
# package, optimizer = ('cvxpy', 'SCS')

criterion = designer_1.d_opt_criterion
result = designer_1.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    optimize_sampling_times=True,
)

designer_1.print_optimal_candidates()

designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities()
designer_1.show_plots()
