import numpy as np

from pydex.core.designer import Designer


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] * np.exp(model_parameters[1] * ti_controls[0])
    ])

if __name__ == '__main__':
    designer = Designer()
    designer.simulate = simulate

    reso = 21j
    tic = np.mgrid[0:0.5:reso]
    designer.ti_controls_candidates = np.array([tic]).T

    designer.model_parameters = [5.5, -5.0]

    designer.initialize(verbose=2)

    """ 
    Pseudo-bayesian type do not really matter in this case because only a single model 
    parameter is involved i.e, information is a scalar, all criterion becomes equivalent to 
    the information matrix itself.
    """

    designer.design_experiment(
        designer.d_opt_criterion,
        write=False,
        package="cvxpy",
        optimizer="MOSEK",
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_controls()

    print("Local Design's Optimal Efforts:")
    print(np.array2string(designer.efforts, separator=", "))
    print("".center(100, "#"))

    designer.show_plots()
