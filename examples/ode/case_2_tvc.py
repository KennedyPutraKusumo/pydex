import numpy as np
from pyomo import dae as pod
from pyomo import environ as po

from pydex.core.designer import Designer


def simulate(model, simulator, ti_controls, tv_controls, sampling_times, model_parameters):
    """ fixing the control variables """
    # time-invariant
    model.theta_0.fix(model_parameters[0])
    model.theta_1.fix(model_parameters[1])
    model.alpha_a.fix(model_parameters[2])
    model.alpha_b.fix(0)
    model.nu.fix(model_parameters[3])

    model.tau.fix(max(sampling_times))
    model.ca[0].fix(ti_controls[0])
    model.cb[0].fix(0)

    """ time-varying controls """
    model.tvc[model.temp] = tv_controls[0]

    """ ensuring pyomo returns state values at given sampling times """
    model.t.initialize = np.array(sampling_times) / model.tau.value
    model.t.order_dict = {}  # to suppress pyomo warnings for duplicate elements
    model.t._constructed = False  # needed so we can re-initialize the continuous set
    model.t._data = {}
    model.t._fe = []
    model.t.value_list = []
    model.t.value = []
    model.t.construct()  # line that re-initializes the continuous set

    """ simulating """
    simulator.simulate(integrator='idas', varying_inputs=model.tvc)
    simulator.initialize_model()

    """" extracting results and returning it in appropriate format """
    normalized_sampling_times = sampling_times / model.tau.value
    ca = np.array([model.ca[t].value for t in normalized_sampling_times])
    cb = np.array([model.cb[t].value for t in normalized_sampling_times])

    return np.array([ca, cb]).T


def create_model():
    """ defining the model """
    model = po.ConcreteModel()
    model.t = pod.ContinuousSet(bounds=(0, 1))
    model.tau = po.Var()

    model.temp = po.Var(model.t)

    model.ca = po.Var(model.t, bounds=(0, 50))
    model.cb = po.Var(model.t, bounds=(0, 50))
    model.dca_dt = pod.DerivativeVar(model.ca, wrt=model.t)
    model.dcb_dt = pod.DerivativeVar(model.cb, wrt=model.t)

    model.theta_0 = po.Var()  # model parameters
    model.theta_1 = po.Var()
    model.alpha_a = po.Var()
    model.alpha_b = po.Var()
    model.nu = po.Var()

    model.tvc = po.Suffix(direction=po.Suffix.LOCAL)

    def _material_balance_a(m, t):
        k = po.exp(m.theta_0 + m.theta_1 * (m.temp[t] - 273.15) / m.temp[t])
        return m.dca_dt[t] / m.tau == - k * (m.ca[t] ** model.alpha_a) * (
                model.cb[t] ** model.alpha_b)

    model.material_balance_a = po.Constraint(model.t, rule=_material_balance_a)

    def _material_balance_b(m, t):
        k = po.exp(m.theta_0 + m.theta_1 * (m.temp[t] - 273.15) / m.temp[t])
        return m.dcb_dt[t] / m.tau == m.nu * k * (m.ca[t] ** model.alpha_a) * (
                model.cb[t] ** model.alpha_b)

    model.material_balance_b = po.Constraint(model.t, rule=_material_balance_b)

    return model


pre_exp_constant = 0.1
activ_energy = 5000
theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
theta_1 = activ_energy / (8.314159 * 273.15)

""" create a pyomo model """
model_1 = create_model()
simulator_1 = pod.Simulator(model_1, package='casadi')

""" create a designer """
designer_1 = Designer()

""" pass pyomo model and simulator to designer """
designer_1.model = model_1
designer_1.simulator = simulator_1

""" overwrite the designer's simulate function """
designer_1.simulate = simulate

""" specifying nominal model parameter """
theta_nom = np.array([theta_0, theta_1, 1, 0.5])  # value of theta_0, theta_1, alpha_a, nu
designer_1.model_parameters = theta_nom  # assigning it to the designer's theta

""" creating experimental candidates, here, it is generated as a grid """
tic, tvc = designer_1.enumerate_candidates(
    bounds=[
        [1, 5],
        [273.15, 323.15],
    ],
    levels=[
        3,
        3,
    ],
    switching_times=np.array([
        None,
        [0, 0.25, 0.50, 0.75],
    ]),
)
spt_candidates = np.array([np.linspace(0, 200, 10) for _ in range(tic.shape[0])])

""" passing the experimental candidates to the designer """
designer_1.ti_controls_candidates = tic
designer_1.tv_controls_candidates = tvc
designer_1.sampling_times_candidates = spt_candidates

""" initializing designer """
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detail

""" (optional) plotting attributes """
designer_1.response_names = ["c_A", "c_B"]
designer_1.model_parameter_names = ["\\theta_0", "\\theta_1", "\\alpha", "\\nu"]

""" D-optimal design """
criterion = designer_1.d_opt_criterion
# criterion = designer_1.a_opt_criterion
# criterion = designer_1.e_opt_criterion

result = designer_1.design_experiment(criterion=criterion, n_spt=2,
                                      optimize_sampling_times=True,
                                      write=False, package="cvxpy")
designer_1.print_optimal_candidates()
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities()
designer_1.show_plots()
