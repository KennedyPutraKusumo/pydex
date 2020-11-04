from pydex.core.designer import Designer
from dow_chemical import simulate
import numpy as np


designer = Designer()
designer.simulate = simulate
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [0.5, 3.5],
        [0, 7],
    ],
    levels=[
        5,
        5,
    ],
)
designer.sampling_times_candidates = np.array([
    np.linspace(0, 1, 21)
    for _ in range(len(designer.ti_controls_candidates))
])
designer.model_parameters = np.array([
    1.8934,
    2.7585,
    1.7540e3,
    6.1894e-3,
    0.0048,
])
designer.initialize(verbose=2)
# designer.estimability_study_fim()
designer.design_experiment(
    criterion=designer.d_opt_criterion,
    write=False,
    optimize_sampling_times=False,
)
designer.print_optimal_candidates()
