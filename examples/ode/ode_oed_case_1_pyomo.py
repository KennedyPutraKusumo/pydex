from pydex.core.designer import Designer
from pyomo import environ as po
from pyomo import dae as pod
import numpy as np


def simulate(model, simulator, ti_controls, tv_controls, model_parameters, sampling_times):
    """ fixing the control variables """
    # time-invariant
    model.beta.fix(model_parameters[0])
    model.tau.fix(max(sampling_times))
    model.ca[0].fix(ti_controls[0])
    # no time-varying control for this example

    """ ensuring pyomo returns state values at given sampling times """
    model.t.initialize = np.array(sampling_times) / model.tau.value
    model.t.order_dict = {}  # to suppress pyomo warnings for duplicate elements
    model.t._constructed = False  # needed so we can re-initialize the continuous set
    model.t._data = {}
    model.t._fe = []
    model.t.value = []
    model.t.value_list = []
    model.t.construct()  # line that re-initializes the continuous set

    """ simulating """
    simulator.simulate(numpoints=100, integrator='idas')
    simulator.initialize_model()

    """" extracting results and returning it in appropriate format """
    normalized_sampling_times = sampling_times / model.tau.value
    ca = np.array([model.ca[t].value for t in normalized_sampling_times])

    return ca

def create_model():
    """ defining the model """
    model = po.ConcreteModel()
    model.t = pod.ContinuousSet(bounds=(0, 1))
    model.tau = po.Var()

    model.ca = po.Var(model.t, bounds=(0, 50))  # set of concentrations in reaction mixture
    model.dca_dt = pod.DerivativeVar(model.ca, wrt=model.t)

    model.beta = po.Var()  # model parameters

    def _material_balance(m, t):
        return m.dca_dt[t] / m.tau == -m.beta * m.ca[t]
    model.material_balance = po.Constraint(model.t, rule=_material_balance)

    return model

def create_simulator(model, package):
    return pod.Simulator(model, package=package)

if __name__ == '__main__':
    """ create a pyomo model """
    model_1 = create_model()

    """ create a designer """
    designer_1 = Designer()

    """ pass pyomo model and simulator to designer """
    designer_1.model = model_1
    designer_1.simulator = create_simulator(model_1, package='casadi')

    """ overwrite the designer's simulate function """
    designer_1.simulate = simulate

    """ specifying nominal model parameter """
    theta_nom = np.array([0.25])  # value of beta, has to be an array
    designer_1.model_parameters = theta_nom  # assigning it to the designer's theta

    """ creating experimental candidates, here, it is generated as a grid """
    n_s_times = 10  # number of equally-spaced sampling time candidates
    n_c = 10  # grid resolution of control candidates generated

    # defining sampling time candidates
    tau_upper = 20
    tau_lower = 0
    # sampling_times_candidates = np.array([np.linspace(tau_lower, tau_upper, n_s_times+_) for _ in range(n_c)])
    sampling_times_candidates = np.array(
        [np.linspace(tau_lower, tau_upper, n_s_times) for _ in range(n_c)])

    # specifying bounds for the grid
    Ca0_lower = 1
    Ca0_upper = 5
    # creating the grid, just some numpy syntax for grid creation
    Ca0_cand = np.mgrid[Ca0_lower:Ca0_upper:complex(0, n_c)]
    Ca0_cand = Ca0_cand.flatten()
    tic_candidates = np.array([Ca0_cand]).T

    """ passing the experimental candidates to the designer """
    designer_1.ti_controls_candidates = tic_candidates
    designer_1.sampling_times_candidates = sampling_times_candidates

    """
    only allow some states to be measurable: 
    as a list or array with column numbers where the measurable states are returned in the simulate function
    optional, if un-specified assume all responses (from simulate function) measurable
    """
    # designer_1.measurable_responses = [1, 2]

    """ initializing designer """
    designer_1.initialize(verbose=1)  # 0: silent, 1: overview, 2: detail

    """ D-optimal continuous design """
    # package, optimizer = ("cvxpy", "MOSEK")
    # package, optimizer = ("cvxpy", "SCS")
    # package, optimizer = ("cvxpy", "CVXOPT")
    package, optimizer = ("scipy", "SLSQP")

    criterion = designer_1.d_opt_criterion
    # criterion = designer_1.a_opt_criterion
    # criterion = designer_1.e_opt_criterion

    d_opt_result = designer_1.design_experiment(criterion=designer_1.d_opt_criterion,
                                                package=package,
                                                optimize_sampling_times=True,
                                                write=False, optimizer=optimizer,
                                                save_sensitivities=True, fd_jac=False)
    designer_1.print_optimal_candidates()
    designer_1.plot_current_design(write=False)
    designer_1.plot_optimal_predictions()
    designer_1.plot_optimal_sensitivities()

    """ an option for saving current designer state """
    designer_1.save_state()

    """ simulate candidates to show model predictions for each candidate """
    # designer_1.simulate_all_candidates(plot_simulation_times=True)
    # designer_1.plot_all_predictions()
