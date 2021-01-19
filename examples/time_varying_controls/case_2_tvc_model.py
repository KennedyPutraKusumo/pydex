from pyomo import dae as pod
from pyomo import environ as po
from matplotlib import pyplot as plt
import numpy as np

def create_model(spt):
    """ defining the model """
    norm_spt = spt / max(spt)

    model = po.ConcreteModel()
    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=norm_spt)
    model.tau = po.Var()

    model.temp = po.Var(model.t, bounds=(200, 400))  # reaction temperature in K

    model.ca = po.Var(model.t, bounds=(0, 50))
    model.cb = po.Var(model.t, bounds=(0, 50))
    model.dca_dt = pod.DerivativeVar(model.ca, wrt=model.t)
    model.dcb_dt = pod.DerivativeVar(model.cb, wrt=model.t)

    model.theta_0 = po.Var()  # model parameters
    model.theta_1 = po.Var()
    model.alpha_a = po.Var()
    model.alpha_b = po.Var()
    model.nu = po.Var()

    model.q_in = po.Var(model.t, bounds=(0, None))  # volumetric flow rate into the reactor in L/min
    model.ca_in = po.Var(model.t, bounds=(0, None))  # molar concentration of A in q_in in mol/L
    model.cb_in = po.Var(model.t, bounds=(0, None))  # molar concentration of B in q_in in mol/L

    model.v = po.Var(model.t, bounds=(0, None))  # volume of reaction mixture in L
    model.dvdt = pod.DerivativeVar(model.v, wrt=model.t)

    model.tvc = po.Suffix(direction=po.Suffix.LOCAL)

    def _material_balance_a(m, t):
        k = po.exp(m.theta_0 + m.theta_1 * (m.temp[t] - 273.15) / m.temp[t])
        return m.dca_dt[t] / m.tau == m.q_in[t] / m.v[t] * (m.ca_in[t] - m.ca[t]) - k * (m.ca[t] ** model.alpha_a) * (
                model.cb[t] ** model.alpha_b)

    model.material_balance_a = po.Constraint(model.t, rule=_material_balance_a)

    def _material_balance_b(m, t):
        k = po.exp(m.theta_0 + m.theta_1 * (m.temp[t] - 273.15) / m.temp[t])
        return m.dcb_dt[t] / m.tau == m.q_in[t] / m.v[t] * (m.cb_in[t] - m.cb[t]) + m.nu * k * (m.ca[t] ** model.alpha_a) * (
                model.cb[t] ** model.alpha_b)

    model.material_balance_b = po.Constraint(model.t, rule=_material_balance_b)

    def _volume_balance(m, t):
        return m.dvdt[t] == m.q_in[t]
    model.volume_balance = po.Constraint(model.t, rule=_volume_balance)

    return model

def simulate(ti_controls, tv_controls, sampling_times, model_parameters):
    tau = np.max(sampling_times)
    model = create_model(sampling_times)

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

    model.v[0].fix(1)

    """ time-varying controls """
    model.tvc[model.temp] = tv_controls[0]
    model.tvc[model.q_in] = tv_controls[1]
    model.tvc[model.ca_in] = tv_controls[2]
    model.tvc[model.cb_in] = tv_controls[3]

    """ simulating """
    simulator = pod.Simulator(model, package="casadi")
    t, profile = simulator.simulate(integrator='idas', varying_inputs=model.tvc)
    if False:
        plt.plot(t, profile)
        plt.show()
    simulator.initialize_model()
    discretizer = po.TransformationFactory("dae.collocation")
    discretizer.apply_to(model, nfe=10, ncp=3, scheme="LAGRANGE-RADAU")

    """" extracting results and returning it in appropriate format """
    normalized_sampling_times = sampling_times / tau
    ca = np.array([model.ca[t].value for t in normalized_sampling_times])
    cb = np.array([model.cb[t].value for t in normalized_sampling_times])

    return np.array([ca, cb]).T

if __name__ == '__main__':
    tic = [5]  # initial A concentration
    tvc = [
        # {0: 200, 0.50: 400},  # reaction temperature
        {0: 300},  # reaction temperature
        {0: 0, 0.5: 1e-2},  # inlet flow rate
        {0: 10},  # ca_in
        {0: 0},  # cb_in
    ]
    mp = [-4.50425, 2.20166, 1.0, 0.5]
    spt = np.linspace(0, 200, 11)
    response = simulate(tic, tvc, spt, mp)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(spt, response[:, 0], label=r"$c_A$")
    axes.plot(spt, response[:, 1], label=r"$c_B$")
    axes.set_xlabel("Time (minutes)")
    axes.set_ylabel("Molar Concentration (M)")
    axes.legend()

    fig.tight_layout()

    plt.show()
