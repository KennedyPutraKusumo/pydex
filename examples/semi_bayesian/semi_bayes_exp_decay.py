import numpy as np

from pydex.core.designer import Designer


def simulate(ti_controls, model_parameters):
    return np.array([
        np.exp(model_parameters[0] * ti_controls[0])
    ])

designer = Designer()
designer.simulate = simulate

reso = 10j
tic = np.mgrid[0:10:reso]
designer.ti_controls_candidates = np.array([tic]).T

np.random.seed(123)
n_scr = 100
designer.model_parameters = np.random.normal(loc=-1, scale=0.05, size=(n_scr, 1))

designer.initialize()

""" 
semi-bayesian type do not really matter in this case because only a single model 
parameter is involved i.e, information is a scalar, all criterion becomes equivalent to 
the information matrix itself.
"""
# sb_type = 0
sb_type = 1

designer.design_experiment(designer.a_opt_criterion, semi_bayes_type=sb_type,
                           write=False)
designer.print_optimal_candidates()
