from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
import numpy as np


def create_model(sampling_times):
    norm_spt = sampling_times / max(sampling_times)
    m = po.ConcreteModel()
    m.t = pod.ContinuousSet(bounds=(0, 1), initialize=norm_spt)
    m.tau = po.Param(initialize=np.max(sampling_times))

    # population compartments
    m.s = po.Var(m.t, bounds=(0, None))
    m.i = po.Var(m.t, bounds=(0, None))
    m.r = po.Var(m.t, bounds=(0, None))

    # differential variables
    m.dsdt = pod.DerivativeVar(m.s, wrt=m.t)
    m.didt = pod.DerivativeVar(m.i, wrt=m.t)
    m.drdt = pod.DerivativeVar(m.r, wrt=m.t)

    # model parameters
    m.beta = po.Var()
    m.gamma = po.Var()

    """ Model Equations """
    def _s_bal(m, t):
        n = m.s[t] + m.i[t] + m.r[t]
        return m.dsdt[t] / m.tau == - m.beta * m.i[t] * m.s[t] / n
    m.s_bal = po.Constraint(m.t, rule=_s_bal)

    def _i_bal(m, t):
        n = m.s[t] + m.i[t] + m.r[t]
        return m.didt[t] / m.tau == m.beta * m.i[t] * m.s[t] / n - m.gamma * m.i[t]
    m.i_bal = po.Constraint(m.t, rule=_i_bal)

    def _r_bal(m, t):
        return m.drdt[t] / m.tau == m.gamma * m.i[t]
    m.r_bal = po.Constraint(m.t, rule=_r_bal)

    return m

def simulate(ti_controls, sampling_times, model_parameters):
    m = create_model(sampling_times)

    """ Fixing Control Variables """
    # initial conditions
    m.s[0].fix(ti_controls[0])
    m.i[0].fix(ti_controls[1])
    m.r[0].fix(ti_controls[2])

    # model parameters
    m.beta.fix(model_parameters[0])
    m.gamma.fix(model_parameters[1])

    simulator = pod.Simulator(m, package="casadi")
    t, profile = simulator.simulate(integrator="idas")
    simulator.initialize_model()

    t = [po.value(tt) for tt in m.t]
    s = [st.value for st in m.s.values()]
    i = [it.value for it in m.i.values()]
    r = [rt.value for rt in m.r.values()]

    y = np.asarray([s, i, r,]).T

    return y

if __name__ == '__main__':
    beta_list = np.linspace(5, 10, 100)
    gamma_list = np.ones_like(beta_list)

    fig = plt.figure()
    axes1 = fig.add_subplot(121)
    tic = [100, 1, 0]
    t = np.linspace(0, 10, 101)
    for beta, gamma in zip(beta_list, gamma_list):
        mp = [beta, gamma]
        y = simulate(
            tic,
            t,
            mp
        )

        axes1.plot(t, y[:, 0], label=f"S @ beta = {beta}", c="r", alpha=0.20)
        axes1.plot(t, y[:, 1], label=f"I @ beta = {beta}", c="g", alpha=0.20)
        axes1.plot(t, y[:, 2], label=f"R @ beta = {beta}", c="b", alpha=0.20)
        axes1.set_xlabel("Time (arbitrary unit)")
        axes1.set_ylabel("Population (arbitrary unit)")

    gamma_list = np.linspace(0.5, 1.5, 100)
    beta_list = np.ones_like(gamma_list) * 7.5
    axes2 = fig.add_subplot(122)
    tic = [100, 1, 0]
    t = np.linspace(0, 10, 101)
    for beta, gamma in zip(beta_list, gamma_list):
        mp = [beta, gamma]
        y = simulate(
            tic,
            t,
            mp
        )

        axes2.plot(t, y[:, 0], label=f"S @ beta = {beta}", c="r", alpha=0.20)
        axes2.plot(t, y[:, 1], label=f"I @ beta = {beta}", c="g", alpha=0.20)
        axes2.plot(t, y[:, 2], label=f"R @ beta = {beta}", c="b", alpha=0.20)
        axes2.set_xlabel("Time (arbitrary unit)")
        axes2.set_ylabel("Population (arbitrary unit)")
    plt.show()
