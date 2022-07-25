from pyomo import environ as po
from pyomo import dae as pod
import numpy as np
import math
from moddex.pyosens import PyosensSimulator

rxn_order = True
tvc_feed = True
fix_switch = True


def create_model_order(spt):
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
    model.alpha_a = po.Var(model.t)
    model.alpha_b = po.Var(model.t)

    model.T = po.Var()  # rxn temperature

    model.N_A0 = po.Var()  # initial A charged (mol)
    model.V_A0 = po.Var()  # initial A charged (L)
    model.N_Bin = po.Var()  # inlet B mol (to define concentration)
    model.V_Bin = po.Var()  # inlet B vol (to define concentration)
    model.N_B0 = po.Var()

    def _mol_bal(m, t):
        R = 8.314
        N_A = m.N_A0 * (1 - m.x[t])
        N_B = m.N_Bin / m.V_Bin * (m.V[t] - m.V_A0) - m.N_A0 * m.x[t] + m.N_B0
        r = m.k[t] * po.exp(-m.ea[t] / (R * m.T)) * N_A ** m.alpha_a[t] * N_B ** m.alpha_b[t] / (m.V[t] ** 2)
        return m.dxdt[t] == m.tau * (r * m.V[t] / m.N_A0)

    model.mol_bal = po.Constraint(model.t, rule=_mol_bal)

    def _vol_bal(m, t):
        return m.dVdt[t] == m.tau * (m.u[t])

    model.vol_bal = po.Constraint(model.t, rule=_vol_bal)

    model.tvc = po.Suffix(direction=po.Suffix.LOCAL)

    return model


def simulate_order(ti_controls, tv_controls, sampling_times, model_parameters, feasibility=False, cooling_failure=False):
    model = create_model_order(sampling_times)

    """ Simulation Parameters """
    model.tvc[model.k] = {0: model_parameters[0]}
    model.tvc[model.ea] = {0: model_parameters[1]}
    if rxn_order:
        alpha_A = model_parameters[2]
        alpha_B = model_parameters[3]
        model.tvc[model.alpha_a] = {0: model_parameters[2]}
        model.tvc[model.alpha_b] = {0: model_parameters[3]}
    else:
        alpha_A = 1
        alpha_B = 1
        model.tvc[model.alpha_a] = {0: alpha_A, 1: alpha_A}
        model.tvc[model.alpha_b] = {0: alpha_B, 1: alpha_B}

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
    if rxn_order:
        model.N_B0.fix(1e-6)
    else:
        model.N_B0.fix(1e-6)

    # time-varying
    if tvc_feed:
        if fix_switch:
            model.tvc[model.u] = tv_controls[0]
        else:
            model.tvc[model.u] = {0: ti_controls[1],
                                  ti_controls[2]: ti_controls[3],
                                  ti_controls[2]+ti_controls[4]: 0,
                                  }
    else:
        model.tvc[model.u] = {0: ti_controls[1], ti_controls[2]: 0}

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
    N_B = (model.N_Bin.value / model.V_Bin.value * (V - model.V_A0.value) - model.N_A0.value * x) + model.N_B0.value
    N_A = ((1 - x) * model.N_A0.value)
    r = (model_parameters[0] * math.exp(-model_parameters[1] / (8.314 * model.T.value))
         * N_A**alpha_A * N_B**alpha_B / (V ** 2))
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
        r_by_N_A = (model_parameters[0] * math.exp(-model_parameters[1] / (8.314 * model.T.value))
                    * N_A**(alpha_A-1) * N_B**alpha_B / (V ** 2))
        r_by_N_B = (model_parameters[0] * math.exp(-model_parameters[1] / (8.314 * model.T.value))
                    * N_A**alpha_A * N_B**(alpha_B-1) / (V ** 2))
        drdk = (r / model_parameters[0] - (alpha_A*r_by_N_A + alpha_B*r_by_N_B) * model.N_A0.value * dxdk)
        dqdk = np.where(drdk != 0.0, drdk * h_rxn * V, 0.0)
        sens2[:, 1, 0] = dqdk[:]
        dxde = sens2[:, 0, 1]
        drdE = (-r / (8.314*model.T.value) - (alpha_A*r_by_N_A + alpha_B*r_by_N_B) * model.N_A0.value * dxde)
        dqdE = np.where(drdE != 0.0, drdE * h_rxn * V, 0.0)
        sens2[:, 1, 1] = dqdE[:]
        if rxn_order:
            dxdalpha_a = sens2[:, 0, 2]
            drdalpha_a = r * np.log(N_A) - (alpha_A*r_by_N_A + alpha_B*r_by_N_B) * model.N_A0.value * dxdalpha_a
            dqdalpha_a = np.where(drdalpha_a != 0.0, drdalpha_a * h_rxn * V, 0.0)
            sens2[:, 1, 2] = dqdalpha_a[:]
            dxdalpha_b = sens2[:, 0, 3]
            drdalpha_b = r * np.log(N_B) - (alpha_A*r_by_N_A + alpha_B*r_by_N_B) * model.N_A0.value * dxdalpha_b
            dqdalpha_b = np.where(drdalpha_b != 0.0, drdalpha_b * h_rxn * V, 0.0)
            sens2[:, 1, 3] = dqdalpha_b[:]
        sens2[:, 0, 0] = 100 * sens2[:, 0, 0]
        sens2[:, 0, 1] = 100 * sens2[:, 0, 1]
        if rxn_order:
            sens2[:, 0, 2] = 100 * sens2[:, 0, 2]
            sens2[:, 0, 3] = 100 * sens2[:, 0, 3]

    if feasibility:
        return np.min(T_max - T_cf)

    if simulator.do_sensitivity_analysis:
        return np.array([
            x,
            q_rx,
        ]).T, sens2
    else:
        if cooling_failure:
            return np.array([
                x,
                q_rx,
                T_cf
            ]).T
        else:
            return np.array([
                x,
                q_rx,
            ]).T
