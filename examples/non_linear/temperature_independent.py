from pyomo import environ as po
from pyomo import dae as pod
from pydex.core.designer import Designer
import numpy as np

def simulate(ti_controls, sampling_times, model_parameters):
    ca = np.array([
        np.exp(-model_parameters[0] * t) for t in sampling_times
    ])
    cb = np.array([
        1 - np.exp(-model_parameters[0] * t) for t in sampling_times
    ])
    return np.array([ca, cb]).T

designer = Designer()
designer.simulate = simulate
designer.ti_controls_candidates = np.ones((1, 1))
designer.model_parameters = np.array([1])
designer.sampling_times_candidates = np.array([np.linspace(0, 10, 1001)])
designer.initialize(verbose=2)
designer.design_experiment(
    designer.d_opt_criterion,
    optimize_sampling_times=True,
    write=False,
    optimizer="MOSEK",
    mosek_params={
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
        "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
    },
)
designer.print_optimal_candidates()
designer.plot_optimal_predictions()
designer.plot_sensitivities()
designer.show_plots()
