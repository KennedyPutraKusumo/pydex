from pyomo import dae as pod
from pyomo import environ as po
from matplotlib import pyplot as plt
import numpy as np


def simulate(ti_controls, sampling_times, model_parameters):
    """ ensuring pyomo returns state values at given sampling times """
    model, simulator = create_model(sampling_times)

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
    model.temp.fix(ti_controls[1])
    # no time-varying control for this example

    """ simulating """
    simulator.simulate(integrator='idas')
    simulator.initialize_model()

    """" extracting results and returning it in appropriate format """
    norm_spt = sampling_times / model.tau.value
    ca = np.array([model.ca[t].value for t in norm_spt])
    cb = np.array([model.cb[t].value for t in norm_spt])

    return np.array([ca, cb]).T


def create_model(spt):
    """ defining the model """
    norm_spt = spt / max(spt)

    model = po.ConcreteModel()
    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=norm_spt)
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

    simulator = pod.Simulator(model, package='casadi')

    return model, simulator


def simulate_tvc(ti_controls, tv_controls, sampling_times, model_parameters):
    tau = np.max(sampling_times)
    normalized_sampling_times = sampling_times / tau
    model = create_model_tvc(normalized_sampling_times)

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

    """ simulating """
    simulator = pod.Simulator(model, package="casadi")
    simulator.simulate(integrator='idas', varying_inputs=model.tvc)
    simulator.initialize_model()
    discretizer = po.TransformationFactory("dae.collocation")
    discretizer.apply_to(model, nfe=10, ncp=3, scheme="LAGRANGE-RADAU")

    """" extracting results and returning it in appropriate format """
    ca = np.array([model.ca[t].value for t in normalized_sampling_times])
    cb = np.array([model.cb[t].value for t in normalized_sampling_times])

    return np.array([ca, cb]).T


def create_model_tvc(spt):
    """ defining the model """

    model = po.ConcreteModel()
    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=spt)
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


if __name__ == '__main__':
    tic = [1, 323.15]

    pre_exp_constant = 0.1
    activ_energy = 5000
    theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
    theta_1 = activ_energy / (8.314159 * 273.15)
    theta_nom = np.array(
        [theta_0, theta_1, 1, 0.1]
    )
    theta_nom = [0.,  1.,  1.1, 0.1]

    spt = np.linspace(0, 200, 11)

    y = simulate(tic, spt, theta_nom)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(
        spt,
        y[:, 0],
        label="$c_A$",
        marker="o",
    )
    axes.plot(
        spt,
        y[:, 1],
        label="$c_B$",
        marker="o",
    )
    axes.legend()
    axes.set_xlabel("Time (min)")
    axes.set_ylabel("Concentration (mol/L)")
    fig.tight_layout()
    plt.show()

    mp_bounds = np.array([
        [-15, 0],
        [0, 5],
        [1, 2],
        [0, 1],
    ])
    reso = 11j
    mp1, mp2, mp3, mp4 = np.mgrid[
        mp_bounds[0][0]:mp_bounds[0][1]:reso,
        mp_bounds[1][0]:mp_bounds[1][1]:reso,
        mp_bounds[2][0]:mp_bounds[2][1]:reso,
        mp_bounds[3][0]:mp_bounds[3][1]:reso,
    ]
    mp1 = mp1.flatten()
    mp2 = mp2.flatten()
    mp3 = mp3.flatten()
    mp4 = mp4.flatten()

    for mp in np.array([mp1, mp2, mp3, mp4]).T:
        try:
            print(f"Model parameters: {mp}")
            y = simulate(tic, spt, mp)
        except RuntimeError:
            print(f"Simulation error at model parameters:")
            print(mp)

    plt.show()
