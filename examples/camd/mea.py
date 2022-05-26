from pydex.core.designer import Designer
import numpy as np


def bp_prediction(ti_controls, model_parameters):
    """
    Functional groups considered
    {NH2CH2, CH2, OH}
    Ordering of the functional groups follow the order reported above

    the ti_control is n;
    a 3 element array that corresponds to the number of {NH2CH2, CH2, OH} groups
    """

    n = ti_controls
    bp = n[0] * model_parameters[0] + n[1] * model_parameters[1] + n[2] * model_parameters[2]
    bp = model_parameters[3] * np.exp(bp)

    return np.array([
        bp,
    ])

# Linear model for BP
if True:
    designer = Designer()
    designer.simulate = bp_prediction

    tic = designer.enumerate_candidates(
        bounds=[
            [0, 2],     # NH2CH2
            [0, 5],     # CH2
            [0, 2],     # OH
        ],
        levels=[
            3,
            6,
            3,
        ],
    )
    """ filter candidates to feasible molecules only """
    # filter out structures with incorrect valencies
    feas_tic = np.array([c for c in tic if np.sum(c[0] * 1 + c[1] * 0 + c[2] * 1 == 2)])
    # filter out non-amine structures
    if True:
        feas_tic = np.array([c for c in feas_tic if c[0] >= 1])
        designer.error_cov = np.diag([100000000000])
    else:
        designer.error_cov = np.diag([1000000000])
    designer.ti_controls_candidates = feas_tic
    designer.model_parameters = np.array([
        2.0, 0.5815, 2.1385, -100,
    ])

    designer.initialize(verbose=2)

    designer._eps = 1e-4
    designer.design_experiment(
        designer.d_opt_criterion,
        regularize_fim=True,
    )
    designer.print_optimal_candidates()
    n_exp = 9
    designer.apportion(
        n_exp,
    )
