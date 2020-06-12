from pydex.core.designer import Designer
from pickle import load
from os import getcwd
import numpy as np

def simulate(ti_controls, model_parameters):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        # interaction term
        model_parameters[3] * ti_controls[0] * ti_controls[1] +
        # squared terms
        model_parameters[4] * ti_controls[0] ** 2 +
        model_parameters[5] * ti_controls[1] ** 2
    ])

designer = Designer()
designer.simulate = simulate

alpha = 0.85
with open(getcwd() + "/ns_output.pkl", "rb") as file:
    ns_output = load(file)
experimental_candidates = ns_output["solution"]["probabilistic_phase"]["samples"]
safe_candidates = np.asarray(experimental_candidates["coordinates"])[np.where(np.asarray(experimental_candidates["phi"]) >= alpha)]
designer.ti_controls_candidates = safe_candidates

designer.model_parameters = np.ones(6)

designer.initialize(verbose=2)

criterion = designer.d_opt_criterion
designer.design_experiment(criterion, write=False)
designer.print_optimal_candidates()

designer.ti_controls_names = [r"$x_1$", r"$x_2$"]
designer.plot_controls(non_opt_candidates=True)
designer.show_plots()
