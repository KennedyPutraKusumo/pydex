from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
import numpy as np


def create_model(spt):

    model = po.ConcreteModel()

    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=spt)
    model.tau = po.Var(bounds=(0, None))

    model.y1 = po.Var(model.t)
    model.y2 = po.Var(model.t)
    model.y3 = po.Var(model.t)
    model.y4 = po.Var(model.t)

    model.x1 = po.Var()
    model.x2 = po.Var()
    model.x3 = po.Var()
    model.x4 = po.Var()

    model.k1        = po.Var()
    model.k2        = po.Var()
    model.k1r       = po.Var()
    model.beta1     = po.Var()
    model.beta2     = po.Var()

    model.dy1dt = pod.DerivativeVar(model.y1, wrt=model.t)
    model.dy2dt = pod.DerivativeVar(model.y2, wrt=model.t)
    model.dy3dt = pod.DerivativeVar(model.y3, wrt=model.t)

    def _ode_1(m, t):
        s = (- 2 * m.x3 + m.x4 + 2 * m.y1[t] - m.y2[t] + m.y3[t]) / (m.y1[t] + m.beta1 * (- m.x3 + m.x4 + m.y1[t] - m.y2[t]) + m.beta2 * m.y3[t])
        return m.dy1dt[t] == m.tau * (- m.k2 * m.y1[t] * m.y2[t] * s)
    model.ode_1 = po.Constraint(model.t, rule=_ode_1)

    def _ode_2(m, t):
        s = (- 2 * m.x3 + m.x4 + 2 * m.y1[t] - m.y2[t] + m.y3[t]) / (m.y1[t] + m.beta1 * (- m.x3 + m.x4 + m.y1[t] - m.y2[t]) + m.beta2 * m.y3[t])
        return m.dy2dt[t] == m.tau * (- m.k1 * m.y2[t] * (m.x2 + 2 * m.x3 - m.x4 - 2 * m.y1[t] + m.y2[t] - m.y3[t]) - m.k2 * m.y1[t] * m.y2[t] * s + m.k1r * m.beta1 * (-m.x3 + m.x4 + m.y1[t] - m.y2[t]) * s)
    model.ode_2 = po.Constraint(model.t, rule=_ode_2)

    def _ode_3(m, t):
        s = (- 2 * m.x3 + m.x4 + 2 * m.y1[t] - m.y2[t] + m.y3[t]) / (m.y1[t] + m.beta1 * (- m.x3 + m.x4 + m.y1[t] - m.y2[t]) + m.beta2 * m.y3[t])
        return m.dy3dt[t] == m.tau * (m.k1 * (m.x3 - m.y1[t] - m.y3[t]) * (m.x2 + 2 * m.x3 - m.x4 - 2 * m.y1[t] + m.y2[t] - m.y3[t]) + m.k2 * m.y1[t] * m.y2[t] * s - 0.5 * m.k1r * m.beta2 * m.y3[t] * s)
    model.ode_3 = po.Constraint(model.t, rule=_ode_3)

    def _ae_1(m, t):
        return m.y4[t] == m.y1[0] - m.y1[t] - m.y3[t]
    model.ae_1 = po.Constraint(model.t, rule=_ae_1)

    return model

def simulate(ti_controls, sampling_times, model_parameters):
    model = create_model(sampling_times)

    # # initial conditions
    # model.y1[0].fix(1.6497)
    # model.y2[0].fix(8.2262)
    # model.y3[0].fix(0)
    # model.y4[0].fix(0)

    # initial conditions
    model.y1[0].fix(ti_controls[0])
    model.y2[0].fix(ti_controls[1])
    model.y3[0].fix(0)
    model.y4[0].fix(0)

    # # controls
    # model.x1.fix(ti_controls[0])
    # model.x2.fix(ti_controls[1])
    # model.x3.fix(ti_controls[2])
    # model.x4.fix(ti_controls[3])

    # constants
    model.x1.fix(340.15)
    model.x2.fix(0.0131)
    model.x3.fix(1.6497)
    model.x4.fix(8.2262)

    # model parameters
    model.k1.fix(model_parameters[0])
    model.k2.fix(model_parameters[1])
    model.k1r.fix(model_parameters[2])
    model.beta1.fix(model_parameters[3])
    model.beta2.fix(model_parameters[4])

    # batch time
    model.tau.fix(300)

    simulator = pod.Simulator(model, package="casadi")
    simulator.simulate()
    simulator.initialize_model()

    # t = po.value(model.tau) * np.array([t for t in model.t])
    y1 = np.array([po.value(model.y1[t]) for t in model.t])
    y2 = np.array([po.value(model.y2[t]) for t in model.t])
    y3 = np.array([po.value(model.y3[t]) for t in model.t])
    y4 = np.array([po.value(model.y4[t]) for t in model.t])

    return np.array([y1, y2, y3, y4]).T

if __name__ == '__main__':
    if False:
        model = create_model(np.linspace(0, 1, 101))
        # initial conditions
        model.y1[0].fix(1.3)
        model.y2[0].fix(8.2262)
        model.y3[0].fix(0)
        model.y4[0].fix(0)

        # constants
        model.x1.fix(340.15)
        model.x2.fix(0.0131)
        model.x3.fix(1.6497)
        model.x4.fix(8.2262)

        # nominal model parameters
        if False:
            model.k1.fix(1.726)
            model.k2.fix(2.312)
            model.k1r.fix(240.5)
            model.beta1.fix(0.006)
            model.beta2.fix(0.004)

        # perturbed model parameters
        if False:
            model.k1.fix(2.0712)
            model.k2.fix(2.7744)
            model.k1r.fix(288.6)
            model.beta1.fix(0.0072)
            model.beta2.fix(0.0048)

        # estimated model parameters after step 7
        if False:
            model.k1.fix(1.8740)
            model.k2.fix(2.7565)
            model.k1r.fix(1.7291e8)
            model.beta1.fix(6.27e-8)
            model.beta2.fix(4.8626e-8)

        # estimated model parameters after step 8 and 11
        if True:
            model.k1.fix(1.8934)
            model.k2.fix(2.7585)
            model.k1r.fix(1.7540e3)
            model.beta1.fix(6.1894e-3)
            model.beta2.fix(0.0048)

        # batch time
        model.tau.fix(300)

        simulator = pod.Simulator(model, package="casadi")
        tsim, profiles = simulator.simulate_pyomo(integrator="idas", numpoints=1000)
        simulator.initialize_model()

        t = po.value(model.tau) * np.array([t for t in model.t])
        y1 = np.array([po.value(model.y1[t]) for t in model.t])
        y2 = np.array([po.value(model.y2[t]) for t in model.t])
        y3 = np.array([po.value(model.y3[t]) for t in model.t])
        y4 = np.array([po.value(model.y4[t]) for t in model.t])

        fig = plt.figure()
        axes1 = fig.add_subplot(121)
        axes1.plot(t, y1, label="$y_1$")
        axes1.plot(t, y4, label="$y_4$")
        axes1.plot(t, y3, label="$y_3$")
        axes1.legend()
        axes1.grid(True)
        axes2 = fig.add_subplot(122)
        axes2.plot(t, y2, label="$y_2$")
        axes2.legend()
        axes2.grid(True)
        plt.show()
    if True:
        mp = np.array([1.8934, 2.7585, 1.7540e3, 6.1894e-3, 0.0048])
        y1s, y2s = np.mgrid[0.5:3.5:11j, 0:8:11j]
        y1s, y2s = y1s.flatten(), y2s.flatten()
        successy1 = []
        successy2 = []
        for y1, y2 in zip(y1s, y2s):
            try:
                y = simulate([y1, y2], np.linspace(0, 1, 21), mp)
                if np.any(y < -1e-5):
                    raise RuntimeError
                successy1.append(y1)
                successy2.append(y2)
                # fig2 = plt.figure()
                # axes2 = fig2.add_subplot(121)
                # axes2.plot(
                #     np.linspace(0, 1, 21),
                #     y[:, (0, 3, 2)]
                # )
                # axes3 = fig2.add_subplot(122)
                # axes3.plot(
                #     np.linspace(0, 1, 21),
                #     y[:, 1],
                # )
                # plt.show()
            except RuntimeError:
                pass
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.scatter(successy1, successy2)
        plt.show()
