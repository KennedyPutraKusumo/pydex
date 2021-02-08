import numpy as np
from pyomo import dae as pod
from pyomo import environ as po

from pydex.core.designer import Designer


def simulate(ti_controls, sampling_times, model_parameters):
    """ ensuring pyomo returns state values at given sampling times """
    normalized_sampling_times = sampling_times / max(sampling_times)
    model, simulator = create_model(normalized_sampling_times)
    model.tau.fix(max(sampling_times))

    """ fixing the control variables """
    # time-invariant
    model.theta_0.fix(model_parameters[0])
    model.theta_1.fix(model_parameters[1])
    model.alpha_a.fix(model_parameters[2])
    model.alpha_b.fix(0)
    model.nu.fix(model_parameters[3])
    # model.nu.fix(1)

    model.ca[0].fix(ti_controls[0])
    model.cb[0].fix(0)
    model.temp.fix(ti_controls[1])
    # no time-varying control for this example

    """ simulating """
    simulator.simulate(integrator='idas')
    simulator.initialize_model()

    """" extracting results and returning it in appropriate format """
    normalized_sampling_times = sampling_times / model.tau.value
    ca = np.array([model.ca[t].value for t in normalized_sampling_times])
    cb = np.array([model.cb[t].value for t in normalized_sampling_times])

    return np.array([ca, cb]).T


def create_model(spt):
    """ defining the model """
    model = po.ConcreteModel()
    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=spt)
    model.tau = po.Var()

    model.temp = po.Var()

    model.ca = po.Var(model.t, bounds=(0, 50))
    model.cb = po.Var(model.t, bounds=(0, 50))
    model.dca_dt = pod.DerivativeVar(model.ca, wrt=model.t)
    model.dcb_dt = pod.DerivativeVar(model.cb, wrt=model.t)

    model.theta_0 = po.Var()  # model parameters
    model.theta_1 = po.Var()
    model.alpha_a = po.Var()
    model.alpha_b = po.Var()
    model.nu = po.Var()

    def _material_balance_a(m, t):
        k = po.exp(m.theta_0 + m.theta_1 * (m.temp - 273.15) / m.temp)
        return m.dca_dt[t] / m.tau == - k * (m.ca[t] ** model.alpha_a) * (
                model.cb[t] ** model.alpha_b)

    model.material_balance_a = po.Constraint(model.t, rule=_material_balance_a)

    def _material_balance_b(m, t):
        k = po.exp(m.theta_0 + m.theta_1 * (m.temp - 273.15) / m.temp)
        return m.dcb_dt[t] / m.tau == m.nu * k * (m.ca[t] ** model.alpha_a) * (
                model.cb[t] ** model.alpha_b)

    model.material_balance_b = po.Constraint(model.t, rule=_material_balance_b)

    simulator = pod.Simulator(model, package="casadi")

    return model, simulator


designer_1 = Designer()
designer_1.simulate = simulate

""" drawing model parameter scenarios from prior """
np.random.seed(123)
n_scr = 10
pre_exp_constant = 0.1
activ_energy = 4000
theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
theta_1 = activ_energy / (8.314159 * 273.15)
np.random.seed(123)  # set a seed for reproducibility
theta_nom = np.array([theta_0, theta_1, 2, 1])  # value of theta_0, theta_1, alpha_a, nu
theta_cov = np.diag(0.20**2 * np.abs(theta_nom))
theta = np.random.multivariate_normal(mean=theta_nom, cov=theta_cov, size=n_scr)
theta[:, 2] = np.round(theta[:, 2])
designer_1.model_parameters = theta  # assigning it to the designer's theta

""" defining control candidates """
tic = designer_1.enumerate_candidates(
    bounds=[
        [1, 5],             # initial C_A concentration
        [273.15, 323.15]    # reaction temperature
    ],
    levels=[
        5,                 # initial C_A concentration
        5,                 # reaction temperature
    ],
)
designer_1.ti_controls_candidates = tic
spt_candidates = np.array([
    np.linspace(0, 200, 11)
    for _ in tic
])
designer_1.sampling_times_candidates = spt_candidates
designer_1.initialize(verbose=2)  # 0: silent, 1: overview, 2: detail

""" (optional) plotting attributes """
designer_1.response_names = ["c_A", "c_B"]
designer_1.model_parameter_names = ["\\theta_0", "\\theta_1", "\\alpha", "\\nu"]

""" Pseudo-bayesian Information Type """
designer_1.solve_cvar_problem(
    designer_1.cvar_d_opt_criterion,
    beta=0.9,
    optimize_sampling_times=True,
    plot=True,
    reso=5,
)
designer_1.plot_pareto_frontier()
designer_1.show_plots()
