if __name__ == '__main__':
    import numpy as np

    N_exp = 3
    reso = 11j

    x1, x2 = np.mgrid[-1:1:reso, -1:1:reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T

    from pydex.core.designer import Designer

    def simulate(ti_controls, model_parameters):
        return np.array([
            model_parameters[0] +
            model_parameters[1] * ti_controls[0] +
            model_parameters[2] * ti_controls[1]
        ])

    designer = Designer()
    designer.simulate = simulate
    designer.ti_controls_candidates = X
    designer.model_parameters = np.ones(3)
    designer.start_logging()
    designer.initialize(verbose=2)
    designer.design_experiment(
        designer.d_opt_criterion,
        n_exp=N_exp,
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_controls(non_opt_candidates=True)
    designer.stop_logging()
    designer.show_plots()
