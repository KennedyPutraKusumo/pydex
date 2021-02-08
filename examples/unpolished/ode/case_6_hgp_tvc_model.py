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
    m.dpdt = pod.DerivativeVar(m.p, wrt=m.t)                        # rate of change of pressure per day

    m.v = po.Var(bounds=(0, 1))                                     # the unknown volume of the reservoir in trillion cubic metres
    m.T = po.Var(bounds=(0, None))                                  # the temperature in the reservoir, assumed isothermal

    m.q = po.Var(m.t, bounds=(0, None))                             # gas flowrate taken out in examoles per day (10^{18} moles per day)
    m.p_vac = po.Var(m.t, bounds=(21.90, 22), initialize=21.95)     # outlet pressure that drive gas extraction in MPa
    m.mu = po.Var(bounds=(0, None))                                 # dynamic viscosity of gas in MPa.day
    m.L = po.Var(bounds=(0, None))                                  # length of pipe in m
    m.R = po.Var(bounds=(0, None))                                  # pipe radius in m
    m.rho = po.Var(bounds=(0, None))                                # density of gas in kg per m3

    m.a = po.Var(m.t, bounds=(0, None))                 # accumulated production in examoles
    m.dadt = pod.DerivativeVar(m.a, wrt=m.t)            # rate of change of accumulated production (equivalent to m.q)

    def _bal(m, t):
        return m.dpdt[t] * m.v / m.tau == - m.q[t] * 8.314 * m.qin
    m.bal = po.Constraint(m.t, rule=_bal)

    def _hagen_poisueille(m, t):
        return m.p[t] - m.p_vac[t] == 8 * m.mu * m.L * m.q[t] / (3.14159 * m.R ** 4)
    m.hagen_poisueille = po.Constraint(m.t, rule=_hagen_poisueille)

    def _compute_accumulated_production(m, t):
        return m.dadt[t] == m.q[t]
    m.compute_accumulated_production = po.Constraint(m.t, rule=_compute_accumulated_production)

    # defining zero accumulated production at the start
    m.a[0].fix(0.0)

    return m

def simulate_hgp(ti_controls, tv_controls, sampling_times, model_parameters):
    m = create_model_hgp(sampling_times)

    m.p[0].fix(22)                          # 22 MPa - figure from first result of googling "pressure in ghawar field"
    m.T.fix(300)                            # 360 Kelvin - taken from (Bruno Stenger; Tony Pham; Nabeel Al-Afaleg; Paul Lawrence GeoArabia (2003) 8 (1): 9–42.)
    m.tau.fix(max(sampling_times))          # hours

    m.v.fix(model_parameters[0])
    m.mu.fix(1000)

    m.L.fix(ti_controls[0])
    m.R.fix(ti_controls[1])

    m.tvc = po.Suffix(direction=po.Suffix.LOCAL)
    m.tvc[m.p_vac] = tv_controls[0]

    simulator = pod.Simulator(m, package="casadi")
    t, profile = simulator.simulate(
        numpoints=101,
        integrator="idas",
        varying_inputs=m.tvc,
    )

    discretizer = po.TransformationFactory("dae.collocation")
    discretizer.apply_to(m, nfe=51, ncp=3)
    discretizer.reduce_collocation_points(m, var=m.p_vac, ncp=1, contset=m.t)
    simulator.initialize_model()

    t = [po.value(t) * max(sampling_times) for t in m.t]
    p = [po.value(m.p[t]) for t in m.t]
    a = [po.value(m.a[t]) for t in m.t]
    q = [po.value(m.q[t]) for t in m.t]

    if True:
        tau = max(sampling_times)
        swt = list(tv_controls[0].keys())

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot(t, p, label="Pressure in Reservoir (MPa)")
        axes.plot(
            [swt[0] * tau, swt[1] * tau, swt[1] * tau, swt[2] * tau, swt[2] * tau, swt[3] * tau, swt[3] * tau, tau],
            [tv_controls[0][0], tv_controls[0][0.0], tv_controls[0][0.25], tv_controls[0][0.25], tv_controls[0][0.50], tv_controls[0][0.50], tv_controls[0][0.75],     tv_controls[0][0.75]],
            label="Outlet Pressure (MPa)"
        )
        axes.set_xlabel("Time (Days)")
        axes.set_ylabel("Reservoir Pressure (MPa)")
        axes.legend()
        fig.tight_layout()

        fig2 = plt.figure()
        axes2 = fig2.add_subplot(111)
        axes2.plot(t, np.asarray(a) * 1e9, label="Accumulated Production")
        axes2.set_xlabel("Time (Days)")
        axes2.set_ylabel(r"Accumulated Production ($10^{9}$ Moles)")
        fig2.tight_layout()

        fig3 = plt.figure()
        axes3 = fig3.add_subplot(111)
        axes3.plot(t, np.asarray(q) * 1e9, label="Gas Flowrate")
        axes3.set_xlabel("Time (Days)")
        axes3.set_ylabel("Gas Flow Rate ($10^9$ Moles per Day)")
        fig3.tight_layout()

        print(f"Productivity: {po.value(m.a[1]) * 1e9 / max(sampling_times)} Billion moles per day")

    norm_spt = sampling_times / max(sampling_times)
    p = [po.value(m.p[t]) for t in norm_spt]
    p = np.asarray(p)[:, None]

    return p

def optimal_extraction(ti_controls, sampling_times, model_parameters):
    m = create_model_hgp(sampling_times)

    def _objective(m):
        # return m.a[1] / max(sampling_times)  # maximum productivity (moles per time)
        return m.a[1]  # maximum production
    m.objective = po.Objective(rule=_objective, sense=po.maximize)

    def _p_vac_cons(m, t):
        return m.p_vac[t] + 0.01 <= m.p[t]
    m.p_vac_cons = po.Constraint(m.t, rule=_p_vac_cons)

    m.p[0].fix(22)                          # 22 MPa - figure from first result of googling "pressure in ghawar field"
    m.T.fix(360)                            # 360 Kelvin - taken from (Bruno Stenger; Tony Pham; Nabeel Al-Afaleg; Paul Lawrence GeoArabia (2003) 8 (1): 9–42.)
    m.tau.fix(max(sampling_times))          # hours

    m.v.fix(model_parameters[0])
    m.mu.fix(model_parameters[1])

    m.L.fix(4500)
    m.R.fix(0.25)

    discretizer = po.TransformationFactory("dae.collocation")
    discretizer.apply_to(m, nfe=51, ncp=3)
    discretizer.reduce_collocation_points(m, var=m.p_vac, ncp=1, contset=m.t)

    solver = po.SolverFactory("ipopt")
    result = solver.solve(m)

    t = [po.value(t) * max(sampling_times) / 24 for t in m.t]
    p = [po.value(m.p[t]) for t in m.t]
    a = [po.value(m.a[t]) for t in m.t]
    q = [po.value(m.q[t]) for t in m.t]
    p_vac = [po.value(m.p_vac[t]) for t in m.t]

    if True:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot(t, p, label="Pressure in Reservoir (MPa)")
        axes.plot(t, p_vac, label="Outlet Pressure (MPa)")
        axes.set_xlabel("Time (Days)")
        axes.set_ylabel("Pressure (MPa)")
        axes.legend()
        fig.tight_layout()

        fig2 = plt.figure()
        axes2 = fig2.add_subplot(111)
        axes2.plot(t, np.asarray(a) * 1e9, label="Accumulated Production")
        axes2.set_xlabel("Time (Days)")
        axes2.set_ylabel(r"Accumulated Production ($10^{9}$ Moles)")
        fig2.tight_layout()

        fig3 = plt.figure()
        axes3 = fig3.add_subplot(111)
        axes3.plot(t, q)
        axes3.set_xlabel("Time (Days)")
        axes3.set_ylabel(r"Gas Flowrate $(10^{12} m^{3} / day)$")
        fig3.tight_layout()

        # fig4 = plt.figure()
        # axes4 = fig4.add_subplot(111)
        # axes4.set_xlabel("Time (Days)")
        # axes4.set_ylabel(r"Outlet Pressure (MPa)")
        # axes4.plot(t, p_vac)
        # fig4.tight_layout()

        print(f"Productivity: {po.value(m.a[1]) * 1e9 / max(sampling_times)} Billion moles per day")

    return t, p, a

if __name__ == '__main__':
    tic = [4500, 1]
    mp = [
        3.1,          # estimated gas in place - taken from wikipedia page of ghawar field on 2020-12-25
        1.28125E-16,  # viscosity in MPa.day - converted from 1.107x10^(-5) Pa.s
    ]
    spt = np.linspace(0, 365, 366)
    # Simulation
    if True:
        tvc = [{
            0.00: 21.995,
            0.25: 21.990,
            0.50: 21.985,
            0.75: 21.980,
        }]
        # tvc = [{
        #     0.00: 22.00,
        #     0.25: 22.00,
        #     0.50: 22.00,
        #     0.75: 22.00,
        # }]
        simulate_hgp(tic, tvc, spt, mp)
    # Maximize Production or Productivity
    if False:
        t, p, a = optimal_extraction(tic, spt, mp)

    # FOR REFERENCE:
    # THE US Produces 107.25 billion moles of natural gas per day
    # Saudi produces 0.9 billion moles of natural gas per day

    plt.show()
