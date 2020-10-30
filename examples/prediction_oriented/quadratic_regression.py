from pydex.core.designer import Designer
import numpy as np


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] +
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[0] ** 2
    ])

designer = Designer()
designer.simulate = simulate
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1],
    ],
    levels=[101],
)
designer.model_parameters = np.ones(3)
designer.initialize(verbose=2)

criteria = [
    designer.dg_opt_criterion, designer.ag_opt_criterion, designer.eg_opt_criterion,
    designer.di_opt_criterion, designer.ai_opt_criterion, designer.ei_opt_criterion,
]
for criterion in criteria:
    designer.design_experiment(
        criterion=criterion,
        write=False,
        package="scipy",
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_controls(title=True)
designer.show_plots()
