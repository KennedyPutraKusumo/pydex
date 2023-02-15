if __name__ == '__main__':
    import numpy as np
     # 8 experimental candidates, 3 responses, 5 model parameters
    A = np.array([np.array([[1,0,0,0,0],
                     [0,3,0,0,0],
                     [0,0,1,0,0]]),
         np.array([[0,0,2,0,0],
                     [0,1,0,0,0],
                     [0,0,0,1,0]]),
         np.array([[0,0,0,2,0],
                     [4,0,0,0,0],
                     [0,0,1,0,0]]),
         np.array([[1,0,0,0,0],
                     [0,0,2,0,0],
                     [0,0,0,0,4]]),
         np.array([[1,0,2,0,0],
                     [0,3,0,1,2],
                     [0,0,1,2,0]]),
         np.array([[0,1,1,1,0],
                     [0,3,0,1,0],
                     [0,0,2,2,0]]),
         np.array([[1,2,0,0,0],
                     [0,3,3,0,5],
                     [1,0,0,2,0]]),
         np.array([[1,0,3,0,1],
                     [0,3,2,0,0],
                     [1,0,0,2,0]])])
    print(A)  # the sensitivity matrix
    A = A[:, None, :, :]
    from pydex.core.designer import Designer

    def sim(ti_controls, model_parameters):
        return np.zeros(shape=(3))

    designer1 = Designer()
    designer1.simulate = sim
    designer1.ti_controls_candidates = np.zeros((8, 1))
    designer1.model_parameters = np.ones(5)
    designer1.initialize(verbose=2)
    designer1.sensitivities = A
    designer1._model_parameters_changed = False
    designer1._candidates_changed = False
    designer1.design_experiment(
        designer1.d_opt_criterion,
    )
    designer1.print_optimal_candidates()
    designer1.design_experiment(
        designer1.d_opt_criterion,
        n_exp=20,
        rtol=1e-4,
    )
    designer1.print_optimal_candidates()
    designer1.show_plots()
