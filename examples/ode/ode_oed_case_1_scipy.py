from pydex.core.designer import Designer
from scipy.integrate import odeint
import numpy as np


def simulate(ti_controls, sampling_times, model_parameters):
    sol = odeint(dca_dt, args=(model_parameters, 1), y0=ti_controls[0], t=sampling_times, tfirst=True)
    ca = sol[:, 0]
    return ca

def dca_dt(t, ca, theta, a):
    dca_dt = - theta[0] * ca
    return dca_dt

if __name__ == '__main__':
    """ create a designer """
    problem_size = []
    designer_1 = Designer()  # without explicit dimension, one extra ODE evaluation

    """ overwrite the designer's simulate function """
    designer_1.simulate = simulate

    """ specifying nominal model parameter """
    theta_nom = np.array([0.25])  # value of beta, has to be an array
    designer_1.model_parameters = theta_nom  # assigning it to the designer's theta

    """ creating experimental candidates, here, it is generated as a grid """
    n_s_times = 10  # number of equally-spaced sampling time candidates
    n_c = 10  # grid resolution of control candidates generated

    # defining the same sampling time candidates for all experimental candidates
    tau_upper = 20
    tau_lower = 0
    sampling_times_candidates = np.array([np.linspace(tau_lower, tau_upper, n_s_times) for _ in range(n_c)])

    # specifying bounds for the grid
    Ca0_lower = 0.1
    Ca0_upper = 5
    # creating the grid, just some numpy syntax for grid creation
    Ca0_cand = np.mgrid[Ca0_lower:Ca0_upper:complex(0, n_c)]
    Ca0_cand = Ca0_cand.flatten()
    tic_candidates = np.array([Ca0_cand]).T

    # there are no time-varying control for this example, so the next line is optional
    tvc_candidates = np.array([{0: 0, 2.5: 10, 7.5: 2} for _ in range(n_c)])  # empty

    """ passing the experimental candidates to the designer """
    designer_1.ti_controls_candidates = tic_candidates
    designer_1.tv_controls_candidates = tvc_candidates
    designer_1.sampling_times_candidates = sampling_times_candidates

    designer_1.initialize(verbose=2)

    """ we can use the designer to get and plot the sensitivities for all experimental candidates """
    sens = designer_1.eval_sensitivities(save_sensitivities=False)
    designer_1.plot_sensitivities()

    """ solve OED problem """
    package, optimizer = ('cvxpy', 'MOSEK')
    # package, optimizer = ('cvxpy', 'SCS')
    # package, optimizer = ('scipy', 'SLSQP')
    d_opt_result = designer_1.design_experiment(criterion=designer_1.d_opt_criterion,
                                                package=package,
                                                optimize_sampling_times=True,
                                                write=False, optimizer=optimizer)
    designer_1.print_optimal_candidates()
