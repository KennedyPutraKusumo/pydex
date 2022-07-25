from pyomo import environ as po
from pyomo import dae as pod
from matplotlib import pyplot as plt
import numpy as np
import math
from moddex.pyosens import PyosensSimulator


def create_model(spt, isothermal=True):
    model = po.ConcreteModel()

    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=spt)
    model.tau = po.Var(bounds=(0, None))  # batch time (s)

    model.i = po.Set(initialize=[
        "A",
        "B",
        "C",
        "D",
    ])

    model.x = po.Var(model.t)  # conversion of A
    model.dxdt = pod.DerivativeVar(model.x, wrt=model.t)

    model.V = po.Var(model.t)  # volume of rxn mixture
    model.dVdt = pod.DerivativeVar(model.V, wrt=model.t)

    model.u = po.Var(model.t)  # inlet feed rate of B (L/s)

    model.k = po.Var(model.t)  # frequency factor of rxn
    model.ea = po.Var(model.t)  # activation energy of rxn

    if isothermal:
        model.T = po.Var()  # rxn temperature
    else:
        model.T = po.Var(model.t)

    model.N_A0 = po.Var()  # initial A charged (mol)
    model.V_A0 = po.Var()  # initial A charged (L)
    model.N_Bin = po.Var()  # inlet B mol (to define concentration)
    model.V_Bin = po.Var()  # inlet B vol (to define concentration)

    def _mol_bal(m, t):
        R = 8.314
        N_A = m.N_A0 * (1 - m.x[t])
        N_B = m.N_Bin / m.V_Bin * (m.V[t] - m.V_A0) - m.N_A0 * m.x[t]
        if isothermal:
            r = m.k[t] * po.exp(-m.ea[t] / (R * m.T)) * N_A * N_B / (m.V[t] ** 2)
        else:
            r = m.k[t] * po.exp(-m.ea[t] / (R * m.T[t])) * N_A * N_B / (m.V[t] ** 2)
        return m.dxdt[t] == m.tau * (r * m.V[t] / m.N_A0)

    model.mol_bal = po.Constraint(model.t, rule=_mol_bal)

    def _vol_bal(m, t):
        return m.dVdt[t] == m.tau * (m.u[t])

    model.vol_bal = po.Constraint(model.t, rule=_vol_bal)

    model.tvc = po.Suffix(direction=po.Suffix.LOCAL)

    return model


def simulate_one_tic(ti_controls, tv_controls, sampling_times, model_parameters, feasibility=False):
    model = create_model(sampling_times)
    model.tvc[model.u] = {0: ti_controls[1], ti_controls[2]: 0}
    return simulate_tic(model, ti_controls, sampling_times, model_parameters, feasibility)


def simulate_tvc_feed(ti_controls, tv_controls, sampling_times, model_parameters, feasibility=False):
    model = create_model(sampling_times)
    model.tvc[model.u] = {0: ti_controls[1], ti_controls[2]: ti_controls[3], ti_controls[2] + ti_controls[4]: 0}
    return simulate_tic(model, ti_controls, sampling_times, model_parameters, feasibility)


def simulate_tic(model, ti_controls, sampling_times, model_parameters, feasibility=False):
    """ Simulation Parameters """
    # model.k.fix(model_parameters[0])        # L/(mol.s)
    # model.ea.fix(model_parameters[1])       # J/mol
    model.tvc[model.k] = {0: model_parameters[0]}
    model.tvc[model.ea] = {0: model_parameters[1]}

    """ Simulation Controls """
    ti_controls_given = [
        2.5,  # N_A0 (mol)
        2.5,  # N_Bin (mol)
        0.32,  # V_A0 (L)
        0.23,  # V_Bin (L)
        30000,  # batch time (s)
    ]
    model.N_A0.fix(ti_controls_given[0])        # mol
    model.N_Bin.fix(ti_controls_given[1])       # mol
    model.V_A0.fix(ti_controls_given[2])        # L
    model.V_Bin.fix(ti_controls_given[3])       # L
    model.tau.fix(ti_controls_given[4])         # seconds
    model.T.fix(ti_controls[0])                 # Kelvin

    """ Initial Conditions """
    model.x[0].fix(0)
    model.V[0].fix(model.V_A0.value)

    simulator = PyosensSimulator(model, package="casadi")
    simulator.detect_sensitivity_analyis = True
    t, profile = simulator.simulate(
        integrator="idas",
        varying_inputs=model.tvc,
        numpoints=sampling_times.shape[0]
    )
    if simulator.do_sensitivity_analysis:
        resp, sens = profile
        resp2 = np.take(resp, np.array([ispt for ispt in range(t.shape[0]) if t[ispt] in sampling_times]), axis=0)
        x = resp2[:, 0]
        V = resp2[:, 1]
        sens2 = np.take(sens, np.array([ispt for ispt in range(t.shape[0]) if t[ispt] in sampling_times]), axis=0)
    else:
        resp = profile
        resp2 = np.take(resp, np.array([ispt for ispt in range(t.shape[0]) if t[ispt] in sampling_times]), axis=0)
        x = resp2[:, 0]
        V = resp2[:, 1]
    simulator.initialize_model()
    N_B = (model.N_Bin.value / model.V_Bin.value * (V - model.V_A0.value) - model.N_A0.value * x)
    N_A = ((1 - x) * model.N_A0.value)
    r = (model_parameters[0] * math.exp(-model_parameters[1] / (8.314 * model.T.value)) * N_A * N_B / (V ** 2))
    x = x * 100
    r_x_V = r * V
    h_rxn = -62500
    q_rx = r * h_rxn * V
    rho = 900 / 1000  # kg/L
    cp = 2000  # J / (kg.K)
    # temperature if cooling failure - MTSR
    T_cf = model.T.value + np.min(np.array([N_A, N_B]).T, axis=1) * (-h_rxn) / (rho * cp * V)
    T_max = 405  # Kelvin
    if simulator.do_sensitivity_analysis:
        dxdk = sens2[:, 0, 0]
        r_by_N_A = (model_parameters[0] * math.exp(-model_parameters[1] / (8.314 * model.T.value)) * N_B / (V ** 2))
        r_by_N_B = (model_parameters[0] * math.exp(-model_parameters[1] / (8.314 * model.T.value)) * N_A / (V ** 2))
        drdk = (r / model_parameters[0] - (r_by_N_A + r_by_N_B) * model.N_A0.value * dxdk)
        dqdk = np.where(drdk != 0.0, drdk * h_rxn * V, 0.0)
        sens2[:, 1, 0] = dqdk[:]
        dxde = sens2[:, 0, 1]
        drdE = (-r / (8.314*model.T.value) - (r_by_N_A + r_by_N_B) * model.N_A0.value * dxde)
        dqdE = np.where(drdE != 0.0, drdE * h_rxn * V, 0.0)
        sens2[:, 1, 1] = dqdE[:]
        sens2[:, 0, 0] = 100 * sens2[:, 0, 0]
        sens2[:, 0, 1] = 100 * sens2[:, 0, 1]

    if feasibility:
        return np.min(T_max - T_cf)

    if simulator.do_sensitivity_analysis:
        return np.array([
            x,
            q_rx,
        ]).T, sens2
    else:
        return np.array([
            x,
            q_rx,
        ]).T


def simulate_tvc(ti_controls, tv_controls, sampling_times, model_parameters, feasibility=False):
    model = create_model(sampling_times, False)

    """ Simulation Parameters """
    model.tvc[model.k] = {0: model_parameters[0]}
    model.tvc[model.ea] = {0: model_parameters[1]}

    """ Simulation Controls """
    ti_controls_given = [
        2.5,  # N_A0 (mol)
        2.5,  # N_Bin (mol)
        0.32,  # V_A0 (L)
        0.23,  # V_Bin (L)
        30000,  # batch time (s)
    ]
    model.N_A0.fix(ti_controls_given[0])        # mol
    model.N_Bin.fix(ti_controls_given[1])       # mol
    model.V_A0.fix(ti_controls_given[2])        # L
    model.V_Bin.fix(ti_controls_given[3])       # L
    model.tau.fix(ti_controls_given[4])         # seconds

    # time-varying
    model.tvc[model.T] = tv_controls[0]
    model.tvc[model.u] = {0: ti_controls[0], ti_controls[1]: 0}

    """ Initial Conditions """
    model.x[0].fix(0)
    model.V[0].fix(model.V_A0.value)

    simulator = PyosensSimulator(model, package="casadi")
    simulator.detect_sensitivity_analyis = True
    t, profile = simulator.simulate(
        integrator="idas",
        varying_inputs=model.tvc,
        numpoints=sampling_times.shape[0]
    )
    if simulator.do_sensitivity_analysis:
        resp, sens = profile
        resp2 = np.take(resp, np.array([ispt for ispt in range(t.shape[0]) if t[ispt] in sampling_times]), axis=0)
        x = resp2[:, 0]
        V = resp2[:, 1]
        sens2 = np.take(sens, np.array([ispt for ispt in range(t.shape[0]) if t[ispt] in sampling_times]), axis=0)
    else:
        resp = profile
        resp2 = np.take(resp, np.array([ispt for ispt in range(t.shape[0]) if t[ispt] in sampling_times]), axis=0)
        x = resp2[:, 0]
        V = resp2[:, 1]
    simulator.initialize_model()
    switching_times = list(tv_controls[0].keys())
    switching_times.append(1.0)
    iswitch = 0
    tval = tv_controls[0][switching_times[iswitch]]
    T = []
    for spt in sampling_times[:-1]:
        if spt < switching_times[iswitch+1]:
            T.append(tval)
        else:
            iswitch = iswitch + 1
            tval = tv_controls[0][switching_times[iswitch]]
            T.append(tval)
    T.append(tval)
    T = np.array(T)
    N_B = (model.N_Bin.value / model.V_Bin.value * (V - model.V_A0.value) - model.N_A0.value * x)
    N_A = ((1 - x) * model.N_A0.value)
    r = (model_parameters[0] * np.exp(-model_parameters[1] / (8.314 * T)) * N_A * N_B / (V ** 2))
    x = x * 100
    r_x_V = r * V
    h_rxn = -62500
    q_rx = r * h_rxn * V
    rho = 900 / 1000  # kg/L
    cp = 2000  # J / (kg.K)
    # temperature if cooling failure - MTSR
    T_cf = T + np.min(np.array([N_A, N_B]).T, axis=1) * (-h_rxn) / (rho * cp * V)
    T_max = 405  # Kelvin
    if simulator.do_sensitivity_analysis:
        dxdk = sens2[:, 0, 0]
        r_by_N_A = (model_parameters[0] * np.exp(-model_parameters[1] / (8.314 * T)) * N_B / (V ** 2))
        r_by_N_B = (model_parameters[0] * np.exp(-model_parameters[1] / (8.314 * T)) * N_A / (V ** 2))
        drdk = (r / model_parameters[0] - (r_by_N_A + r_by_N_B) * model.N_A0.value * dxdk)
        dqdk = np.where(drdk != 0.0, drdk * h_rxn * V, 0.0)
        sens2[:, 1, 0] = dqdk[:]
        dxde = sens2[:, 0, 1]
        drdE = (-r / (8.314*T) - (r_by_N_A + r_by_N_B) * model.N_A0.value * dxde)
        dqdE = np.where(drdE != 0.0, drdE * h_rxn * V, 0.0)
        sens2[:, 1, 1] = dqdE[:]
        sens2[:, 0, 0] = 100 * sens2[:, 0, 0]
        sens2[:, 0, 1] = 100 * sens2[:, 0, 1]

    if feasibility:
        return np.min(T_max - T_cf)

    if simulator.do_sensitivity_analysis:
        return np.array([
            x,
            q_rx,
        ]).T, sens2
    else:
        return np.array([
            x,
            q_rx,
        ]).T


def simulate(ti_controls, tv_controls, sampling_times, model_parameters, plot=False):
    model = create_model(sampling_times)

    """ Simulation Parameters """
    model.k.fix(model_parameters[0])        # L/(mol.s)
    model.ea.fix(model_parameters[1])       # J/mol

    """ Simulation Controls """
    model.N_A0.fix(ti_controls[0])          # mol
    model.N_Bin.fix(ti_controls[1])         # mol
    model.V_A0.fix(ti_controls[2])          # L
    model.V_Bin.fix(ti_controls[3])         # L
    model.tau.fix(ti_controls[4])           # seconds
    model.T.fix(ti_controls[5])             # Kelvin

    # time-varying
    model.tvc[model.u] = tv_controls[0]

    """ Initial Conditions """
    model.x[0].fix(0)
    model.V[0].fix(model.V_A0.value)

    simulator = pod.Simulator(model, package="casadi")
    t, profile = simulator.simulate(
        integrator="idas",
        varying_inputs=model.tvc,
    )
    discretizer = po.TransformationFactory("dae.collocation")
    discretizer.apply_to(
        model,
        nfe=20,
        ncp=3,
        scheme="LAGRANGE-RADAU",
    )
    simulator.initialize_model()

    t = np.array([t * model.tau.value for t in sampling_times]) / 3600
    u = np.array([model.u[t].value for t in sampling_times])
    x = [model.x[t].value * 100 for t in sampling_times]
    V = np.array([model.V[t].value for t in sampling_times])
    N_B = [po.value(
        model.N_Bin / model.V_Bin * (model.V[t] - model.V_A0) - model.N_A0 * model.x[t])
           for t in sampling_times]
    N_A = [po.value((1 - model.x[t]) * model.N_A0) for t in sampling_times]
    r = [po.value(model.k * po.exp(-model.ea / (8.314 * model.T)) * (model.N_A0 * (1 - model.x[t])) * (model.N_Bin / model.V_Bin * (model.V[t] - model.V_A0) - model.N_A0 * model.x[t]) / (model.V[t] ** 2)) for t in sampling_times]
    r_x_V = np.array(r) * np.array(V)
    h_rxn = -62500
    q_rx = np.array(r) * h_rxn * V
    rho = 900 / 1000  # kg/L
    cp = 2000  # J / (kg.K)
    # temperature if cooling failure - MTSR
    T_cf = model.T.value + np.min(np.array([N_A, N_B]).T, axis=1) * (-h_rxn) / (
                rho * cp * V)
    T_max = 405  # Kelvin

    if plot:
        fig1 = plt.figure(figsize=(15, 9))
        axes1 = fig1.add_subplot(331)
        axes1.plot(
            t,
            x,
            label="Conversion of A",
        )
        axes1.set_xlabel("Time (hour)")
        axes1.set_ylabel("Conversion of A (%)")

        axes2 = fig1.add_subplot(332)
        axes2.plot(
            t,
            q_rx,
            label="Power generated by Reaction",
        )
        axes2.set_xlabel("Time (hour)")
        axes2.set_ylabel("Power generated by Reaction (J/s)")

        axes3 = fig1.add_subplot(333)
        axes3.plot(
            t,
            V,
            label="Volume",
        )
        axes3.set_xlabel("Time (hour)")
        axes3.set_ylabel("Volume (L)")

        axes4 = fig1.add_subplot(334)
        axes4.plot(
            t,
            N_B,
            label="Amount of Species B",
        )
        axes4.set_xlabel("Time (hour)")
        axes4.set_ylabel("Amount of Species B (mol)")

        axes5 = fig1.add_subplot(335)
        axes5.plot(
            t,
            N_A,
            label="Amount of Species A",
        )
        axes5.set_xlabel("Time (hour)")
        axes5.set_ylabel("Amount of Species A (mol)")

        axes6 = fig1.add_subplot(336)
        axes6.plot(
            t,
            r_x_V * 3600,
            label="Reaction rate",
        )
        axes6.set_xlabel("Time (hour)")
        axes6.set_ylabel("Reaction Rate (mol/hour)")

        axes7 = fig1.add_subplot(337)
        axes7.plot(
            t,
            T_cf,
            label="Maximum Temperature of Synthesis Reaction (MTSR)",
        )
        axes7.axhline(
            y=T_max,
            xmin=0,
            xmax=1,
            c="tab:red",
            ls="dashed",
            label=f"Maximum Allowable Temperature ({T_max} Kelvin)",
        )
        axes7.set_xlabel("Time (hour)")
        axes7.set_ylabel("MTSR (Kelvin)")
        axes7.legend()

        fig1.tight_layout()

        plt.show()

    return np.array([
        x,
        q_rx,
    ])


if __name__ == '__main__':
    model_parameters = [
        3.5e11 / 3600,      # frequency factor k (L/(mol.s))
        82500,              # activation energy (J/mol)
    ]
    ti_controls = [
        2.5,                # N_A0 (mol)
        2.5,                # N_Bin (mol)
        0.32,               # V_A0 (L)
        0.23,               # V_Bin (L)
        30000,              # batch time (s)
        70 + 273.15,        # temperature (Kelvin)
    ]
    tv_controls = [
        {
            0: 3e-5,
            # 0.09: 0.7e-5,
            0.09: 0,
        },
    ]
    sampling_times = np.linspace(0, 1, 21)

    simulate(
        ti_controls=ti_controls,
        tv_controls=tv_controls,
        sampling_times=sampling_times,
        model_parameters=model_parameters,
        plot=True,
    )
    ti_controls = [70 + 273.15]
    tv_controls = [
        {
            0: 3e-5,
            # 0.09: 0.7e-5,
            0.09: 0,
        },
    ]

    test = simulate_one_tic(ti_controls, tv_controls, sampling_times, model_parameters)
    print(test)
