from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
import numpy as np

def create_model_hgp(spt):
    m = po.ConcreteModel()

    tau = max(spt)
    norm_spt = spt / tau
    m.t = pod.ContinuousSet(bounds=(0, 1), initialize=norm_spt)     # normalized time variable (unitless)
    m.tau = po.Var(bounds=(0, None))                                # batch time in hours

    m.p = po.Var(m.t, bounds=(0, None))                             # pressure in reservoir in MPa
    m.dpdt = pod.DerivativeVar(m.p, wrt=m.t)                        # rate of change of pressure per hour

    m.v = po.Var(bounds=(0, 1))                                     # the unknown volume of the reservoir in trillion cubic metres
    m.T = po.Var(bounds=(0, None))                                  # the temperature in the reservoir, assumed isothermal

    m.q = po.Var(m.t, bounds=(0, None))                             # gas flowrate taken out in unit volume per unit time
    m.p_vac = po.Var(bounds=(1, None))                              # outlet vacuum to drive gas extraction in unit pressure
    m.mu = po.Var(bounds=(0, None))                                 # dynamic viscosity of gas in unit pressure times unit time
    m.L = po.Var(bounds=(0, None))                                  # length of pipe in unit length
    m.R = po.Var(bounds=(0, None))                                  # pipe radius in unit length
    m.rho = po.Var(bounds=(0, None))                                # density of gas in kg per m3

    def _bal(m, t):
        return m.dpdt[t] * m.v / m.tau == - m.q[t] * 8.314 * m.T
    m.bal = po.Constraint(m.t, rule=_bal)

    def _hagen_poisueille(m, t):
        return m.p[t] - m.p_vac == 8 * m.mu * m.L * m.q[t] / (3.14159 * m.R ** 4)
    m.hagen_poisueille = po.Constraint(m.t, rule=_hagen_poisueille)

    return m

def simulate_hgp(ti_controls, sampling_times, model_parameters):
    m = create_model_hgp(sampling_times)

    m.p[0].fix(22)                          # 22 MPa - figure from first result of googling "pressure in ghawar field"
    m.T.fix(300)                            # 360 Kelvin - taken from (Bruno Stenger; Tony Pham; Nabeel Al-Afaleg; Paul Lawrence GeoArabia (2003) 8 (1): 9–42.)
    m.tau.fix(max(sampling_times))          # hours

    m.v.fix(model_parameters[0])
    # m.mu.fix(model_parameters[1])
    m.mu.fix(1000)

    m.p_vac.fix(ti_controls[0])
    m.L.fix(ti_controls[1])
    m.R.fix(ti_controls[2])

    # m.tvc = po.Suffix(direction=po.Suffix.LOCAL)

    simulator = pod.Simulator(m, package="casadi")
    t, profile = simulator.simulate(
        numpoints=101,
        integrator="idas",
        # varying_inputs=m.tvc,
    )

    simulator.initialize_model()

    t = [po.value(t) * max(sampling_times) / 24 for t in m.t]
    p = [po.value(m.p[t]) for t in m.t]

    if False:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot(t, p)
        axes.set_xlabel("Time (Days)")
        axes.set_ylabel("Reservoir Pressure (MPa)")
        fig.tight_layout()

    p = np.asarray(p)[:, None]

    return p


if __name__ == '__main__':

    if False:
        tvc = [{
            0.00: 0,
            0.25: 1,
            0.50: 0,
            0.75: 1,
        }]
    if True:
        tic = [7, 100, 1]
    mp = [
        3.1,            # estimated gas in place - taken from wikipedia page of ghawar field on 2020-12-25
        # 3.075E-15,      # viscosity in MPa.hr - converted from 1.107x10^(-5) Pa.s
        # 1000,  # viscosity in MPa.hr - converted from 1.107x10^(-5) Pa.s
    ]
    spt = np.linspace(0, 24 * 30, 101)
    simulate_hgp(tic, spt, mp)
    plt.show()
