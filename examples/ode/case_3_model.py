from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
from time import time
import numpy as np

def create_model(spt):
    model = po.ConcreteModel()

    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=spt)
    model.tau = po.Var(bounds=(0, None))

    model.cA = po.Var(model.t, bounds=(0, None))
    model.cB = po.Var(model.t, bounds=(0, None))

    model.dcAdt = pod.DerivativeVar(model.cA, wrt=model.t)
    model.dcBdt = pod.DerivativeVar(model.cB, wrt=model.t)

    model.nu = po.Var(bounds=(0, None))
    model.alpha = po.Var(bounds=(0, None))
    model.beta = po.Var(bounds=(0, None))

    model.T = po.Var(bounds=(0, None))

    model.theta_10 = po.Var()
    model.theta_11 = po.Var()
    model.theta_20 = po.Var()
    model.theta_21 = po.Var()
    model.theta_30 = po.Var()
    model.theta_31 = po.Var()

    def _A_mol_bal(m, t):
        k1 = po.exp(m.theta_10 + m.theta_11 * (m.T - 273.15) / m.T)
        k2 = po.exp(m.theta_20 + m.theta_21 * (m.T - 273.15) / m.T)
        k3 = po.exp(m.theta_30 + m.theta_31 * (m.T - 273.15) / m.T)
        r = k1 * m.cA[t] ** m.alpha / (k2 + k3 * m.cA[t] ** m.beta)
        return m.dcAdt[t] == m.tau * - r
    model.A_mol_bal = po.Constraint(model.t, rule=_A_mol_bal)

    def _B_mol_bal(m, t):
        k1 = po.exp(m.theta_10 + m.theta_11 * (m.T - 273.15) / m.T)
        k2 = po.exp(m.theta_20 + m.theta_21 * (m.T - 273.15) / m.T)
        k3 = po.exp(m.theta_30 + m.theta_31 * (m.T - 273.15) / m.T)
        r = k1 * m.cA[t] ** m.alpha / (k2 + k3 * m.cA[t] ** m.beta)
        return m.dcBdt[t] == m.tau * m.nu * r
    model.B_mol_bal = po.Constraint(model.t, rule=_B_mol_bal)
    simulator = pod.Simulator(model, package="casadi")

    return model, simulator

def simulate(ti_controls, sampling_times, model_parameters):
    model, simulator = create_model(sampling_times)

    # model parameters
    model.theta_10.fix(model_parameters[0])
    model.theta_11.fix(model_parameters[1])
    model.theta_20.fix(model_parameters[2])
    model.theta_21.fix(model_parameters[3])
    model.theta_30.fix(model_parameters[4])
    model.theta_31.fix(model_parameters[5])

    model.nu.fix(model_parameters[6])

    model.alpha.fix(model_parameters[7])
    model.beta.fix(model_parameters[8])

    # initial conditions
    model.cA[0].fix(ti_controls[0])
    model.cB[0].fix(0)

    # experimental controls
    model.T.fix(ti_controls[1])
    model.tau.fix(ti_controls[2])

    t_sim, profile = simulator.simulate(integrator="idas", numpoints=100)
    simulator.initialize_model()

    cA = [po.value(model.cA[t]) for t in model.t]
    cB = [po.value(model.cB[t]) for t in model.t]

    return np.array([cA, cB]).T


if __name__ == '__main__':
    times = []
    start = time()
    tic = [10, 303.15, 10]
    mp = [5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7/3, 3, 5]
    spt = np.linspace(0, 1, 201)

    c = simulate(tic, spt, mp)
    cA, cB = c[:, 0], c[:, 1]
    print(f"One simulation took {time() - start} CPU seconds.")

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(spt, cA, label="$c_A$")
    axes.plot(spt, cB, label="$c_B$")
    axes.set_xlabel("Time (units)")
    axes.set_ylabel("Concentration (mol/L)")
    axes.legend()
    plt.show()

    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    # axes.plot(
    #     np.arange(0, 1000, 1),
    #     times,
    # )
    # plt.show()
