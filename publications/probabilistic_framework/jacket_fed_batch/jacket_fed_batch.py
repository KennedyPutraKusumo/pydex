from pyomo import dae as pod
from pyomo import environ as po
from matplotlib import pyplot as plt
import numpy as np
from moddex.pyosens import PyosensSimulator


def create_model(spt):
    """ defining the model """
    norm_spt = spt / max(spt)

    model = po.ConcreteModel()
    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=norm_spt)
    model.tau = po.Var()

    model.theta_0 = po.Var(model.t)  # model parameters
    model.theta_1 = po.Var(model.t)
    model.alpha_a = po.Var(model.t)
    model.alpha_b = po.Var()
    model.nu = po.Var(model.t)

    model.v = po.Var(model.t, bounds=(0, None))  # volume of reaction mixture in L
    model.dvdt = pod.DerivativeVar(model.v, wrt=model.t)

    model.T = po.Var(model.t, bounds=(0, None))  # reaction temperature in K

    model.ca = po.Var(model.t, bounds=(0, None))
    model.cb = po.Var(model.t, bounds=(0, None))
    model.cc = po.Var(model.t, bounds=(0, None))  # solvent concentration
    model.dca_dt = pod.DerivativeVar(model.ca, wrt=model.t)
    model.dcb_dt = pod.DerivativeVar(model.cb, wrt=model.t)
    model.dcc_dt = pod.DerivativeVar(model.cc, wrt=model.t)

    model.q_in = po.Var(model.t, bounds=(
    0, None))  # volumetric flow rate into the reactor in L/min
    model.q_out = po.Var(model.t, bounds=(
    0, None))  # volumetric flow rate out of the reactor in L/min
    model.ca_in = po.Var(model.t,
                         bounds=(0, None))  # molar concentration of A in q_in in mol/L
    model.cb_in = po.Var(model.t,
                         bounds=(0, None))  # molar concentration of B in q_in in mol/L
    model.cc_in = po.Var(model.t, bounds=(
    0, None))  # molar concentration of C (solvent) in q_in in mol/L

    model.tvc = po.Suffix(direction=po.Suffix.LOCAL)

    model.t_ref = po.Var()

    def _material_balance_a(m, t):
        k = po.exp(m.theta_0[t] + m.theta_1[t] * (m.T[t] - m.t_ref) / m.T[t])
        return m.dca_dt[t] / m.tau == m.q_in[t] / m.v[t] * (m.ca_in[t] - m.ca[t]) - k * (
                    m.ca[t] ** model.alpha_a[t]) * (model.cb[t] ** model.alpha_b)

    model.material_balance_a = po.Constraint(model.t, rule=_material_balance_a)

    def _material_balance_b(m, t):
        k = po.exp(m.theta_0[t] + m.theta_1[t] * (m.T[t] - m.t_ref) / m.T[t])
        return m.dcb_dt[t] / m.tau == m.q_in[t] / m.v[t] * (
                    m.cb_in[t] - m.cb[t]) + m.nu[t] * k * (m.ca[t] ** model.alpha_a[t]) * (
                       model.cb[t] ** model.alpha_b)

    model.material_balance_b = po.Constraint(model.t, rule=_material_balance_b)

    def _material_balance_c(m, t):
        return m.dcc_dt[t] / m.tau == 0
        # return m.dcc_dt[t] / m.tau == m.q_in[t] / m.v[t] * (m.cc_in[t] - m.cc[t])

    model.material_balance_c = po.Constraint(model.t, rule=_material_balance_c)

    def _volume_balance(m, t):
        return m.dvdt[t] / m.tau == m.q_in[t] - m.q_out[t]

    model.volume_balance = po.Constraint(model.t, rule=_volume_balance)

    model.hf_a = po.Var()
    model.hf_b = po.Var()
    model.hf_c = po.Var()

    model.cp_a = po.Var()
    model.cp_b = po.Var()
    model.cp_c = po.Var()

    model.T_in = po.Var(model.t, bounds=(0, None))
    model.Tj = po.Var(model.t, bounds=(0, None))

    model.U = po.Var(model.t)
    model.A = po.Var()

    model.dTdt = pod.DerivativeVar(model.T, wrt=model.t)

    def _rm_heat_balance(m, t):
        h_in_a = m.ca_in[t] * (m.hf_a + m.cp_a * (m.T_in[t] - m.t_ref))
        h_in_b = m.cb_in[t] * (m.hf_b + m.cp_b * (m.T_in[t] - m.t_ref))
        h_in_c = m.cc_in[t] * (m.hf_c + m.cp_c * (m.T_in[t] - m.t_ref))
        h_in = h_in_a + h_in_b + h_in_c

        h_rm_a = m.ca[t] * (m.hf_a + m.cp_a * (m.T[t] - m.t_ref))
        h_rm_b = m.cb[t] * (m.hf_b + m.cp_b * (m.T[t] - m.t_ref))
        h_rm_c = m.cc[t] * (m.hf_c + m.cp_c * (m.T[t] - m.t_ref))
        h_rm = h_rm_a + h_rm_b + h_rm_c

        Q_j = m.U[t] * m.A * (m.T[t] - m.Tj[t])

        k = po.exp(m.theta_0[t] + m.theta_1[t] * (m.T[t] - m.t_ref) / m.T[t])
        dcadt = m.q_in[t] / m.v[t] * (m.ca_in[t] - m.ca[t]) - k * (
                    m.ca[t] ** model.alpha_a[t]) * (model.cb[t] ** model.alpha_b)
        dcbdt = m.q_in[t] / m.v[t] * (m.cb_in[t] - m.cb[t]) + m.nu[t] * k * (
                    m.ca[t] ** model.alpha_a[t]) * (
                        model.cb[t] ** model.alpha_b)
        dccdt = m.q_in[t] / m.v[t] * (m.cc_in[t] - m.cc[t])

        return m.dTdt[t] / m.tau == 1 / (m.cp_a + m.cp_b + m.cp_c) * (
                    m.q_in[t] / m.v[t] * (h_in - h_rm) - Q_j / m.v[t] - (
                        dcadt * (m.hf_a + m.cp_a * (m.T[t] - m.t_ref))) - (
                                dcbdt * (m.hf_b + m.cp_b * (m.T[t] - m.t_ref))) - (
                                dccdt * (m.hf_c + m.cp_c * (m.T[t] - m.t_ref))))

    model.rm_heat_balance = po.Constraint(model.t, rule=_rm_heat_balance)

    model.dTjdt = pod.DerivativeVar(model.Tj, wrt=model.t)
    model.vj = po.Var()
    model.cp_w = po.Var()
    model.q_w = po.Var(model.t, bounds=(0, None))
    model.hf_w = po.Var()
    model.Tw_in = po.Var(model.t, bounds=(0, None))
    model.m_w = po.Var()
    model.rho_w = po.Var()

    def _jacket_heat_bal(m, t):
        hw_in = m.hf_w + m.cp_w * (m.Tw_in[t] - m.t_ref)
        hw_out = m.hf_w + m.cp_w * (m.Tj[t] - m.t_ref)
        Q_j = m.U[t] * m.A * (m.T[t] - m.Tj[t])
        return m.dTjdt[t] / m.tau == 1 / (m.vj * m.cp_w) * (m.q_w[t] * (hw_in - hw_out) + Q_j * m.m_w / m.rho_w)
    model.jacket_heat_bal = po.Constraint(model.t, rule=_jacket_heat_bal)

    return model

def simulate(ti_controls, tv_controls, sampling_times, model_parameters):
    tau = np.max(sampling_times)
    model = create_model(sampling_times)
    norm_tvc = []
    for tvc in tv_controls:
        single_ntvc = {}
        for key, value in tvc.items():
            single_ntvc[key / tau] = value
        norm_tvc.append(single_ntvc)

    model.tau.fix(max(sampling_times))

    model.theta_0.fix(model_parameters[0])
    model.theta_1.fix(model_parameters[1])
    model.alpha_a.fix(model_parameters[2])
    model.alpha_b.fix(0)
    model.nu.fix(model_parameters[3])

    v0 = 2.00  # L
    model.v[0].fix(v0)  # L

    rho_c = 889  # g/L
    m_c = 72.1057  # g/mol
    cc0 = v0 * rho_c / m_c

    model.ca[0].fix(ti_controls[0])
    model.cb[0].fix(0)  # mol/L
    model.cc[0].fix(cc0)

    """ time-varying controls """
    model.tvc[model.q_in] = norm_tvc[0]
    model.tvc[model.q_out] = norm_tvc[1]
    model.tvc[model.ca_in] = {0: 0.5}
    model.tvc[model.cb_in] = {0: 0}
    model.tvc[model.cc_in] = {0: cc0}

    """ rm heat balance variables """
    model.T[0].fix(298.15)
    model.t_ref.fix(273.15)

    model.hf_a.fix(-80000)
    model.hf_b.fix(-180000)
    model.hf_c.fix(-123000)

    model.cp_a.fix(112.4)
    model.cp_b.fix(120)
    model.cp_c.fix(130)

    model.tvc[model.T_in] = {0: 273.15 + 25}

    model.U.fix(model_parameters[4])
    model.A.fix(5)

    """ jacket heat balance variables """
    model.Tj[0].fix(273.15)
    model.vj.fix(2)  # L
    model.cp_w.fix(75.38)  # J.mol-1.K-1
    model.tvc[model.q_w] = norm_tvc[2]
    model.hf_w.fix(-285830)  # J/mol
    model.tvc[model.Tw_in] = {0: 273.15}

    model.m_w.fix(18)  # g/mol
    model.rho_w.fix(1000)  # g/L

    """ simulating """
    simulator = pod.Simulator(model, package="casadi")
    try:
        t, profile = simulator.simulate(
            integrator='idas',
            varying_inputs=model.tvc,
            numpoints=len(sampling_times)+1,
        )
    except RuntimeError:
        print("Model Parameters:")
        print(po.value(model.theta_0))
        print(po.value(model.theta_1))
        print(po.value(model.alpha_a))
        print(po.value(model.nu))

        print("Time-invariant Controls:")
        print(ti_controls)
        print("Time-varying Controls:")
        print(tv_controls)
        print("Sampling Time Candidates:")
        print(sampling_times)
        raise RuntimeError
    if False:
        plt.plot(t, profile)
        plt.show()
    simulator.initialize_model()
    discretizer = po.TransformationFactory("dae.collocation")
    discretizer.apply_to(model, nfe=len(sampling_times), ncp=3, scheme="LAGRANGE-RADAU")

    """" extracting results and returning it in appropriate format """
    normalized_sampling_times = sampling_times / tau
    ca = np.array([model.ca[t].value for t in normalized_sampling_times])
    cb = np.array([model.cb[t].value for t in normalized_sampling_times])
    cc = np.array([model.cc[t].value for t in normalized_sampling_times])
    volume = np.array([model.v[t].value for t in normalized_sampling_times])
    reactor_temp = np.array([model.T[t].value for t in normalized_sampling_times])
    jacket_temp = np.array([model.Tj[t].value for t in normalized_sampling_times])
    rxn_rate = np.array([po.value(po.exp(model.theta_0 + model.theta_1 * (model.T[t] - model.t_ref) / model.T[t]) * (model.ca[t] ** model.alpha_a) * (model.cb[t] ** model.alpha_b)) for t in normalized_sampling_times])

    return np.array([ca, cb, reactor_temp, jacket_temp, volume, rxn_rate]).T
    # return np.array([ca, cb, reactor_temp, jacket_temp, volume, rxn_rate, cc]).T

def simulate2(ti_controls, sampling_times, model_parameters):
    tau = np.max(sampling_times)
    model = create_model(sampling_times)

    model.tau.fix(max(sampling_times))

    model.tvc[model.theta_0] = {0: model_parameters[0]}
    model.tvc[model.theta_1] = {0: model_parameters[1]}
    model.tvc[model.alpha_a] = {0: model_parameters[2]}
    model.alpha_b.fix(0)
    model.tvc[model.nu] = {0: model_parameters[3]}

    v0 = 2.00  # L
    model.v[0].fix(v0)  # L

    rho_c = 889  # g/L
    m_c = 72.1057  # g/mol
    cc0 = v0 * rho_c / m_c

    model.ca[0].fix(0.03)
    model.cb[0].fix(0)  # mol/L
    model.cc[0].fix(cc0)

    """ time-varying controls """
    model.tvc[model.q_in] = {0: 0.50, 100 / tau: ti_controls[0], (100+ti_controls[1]) / tau: 0.50}
    model.tvc[model.q_out] = {0: 0.50, 100 / tau: ti_controls[0], (100+ti_controls[1]) / tau: 0.50}
    model.tvc[model.ca_in] = {0: 0.5, 1: 0.5}
    model.tvc[model.cb_in] = {0: 0, 1: 0}
    model.tvc[model.cc_in] = {0: cc0, 1: cc0}

    """ rm heat balance variables """
    model.T[0].fix(298.15)
    model.t_ref.fix(273.15)

    model.hf_a.fix(-80000)
    model.hf_b.fix(-180000)
    model.hf_c.fix(-123000)

    model.cp_a.fix(112.4)
    model.cp_b.fix(120)
    model.cp_c.fix(130)

    model.tvc[model.T_in] = {0: 273.15 + 25, 1: 273.15 + 25}

    model.tvc[model.U] = {0: model_parameters[4]}
    model.A.fix(5)

    """ jacket heat balance variables """
    model.Tj[0].fix(273.15)
    model.vj.fix(2)  # L
    model.cp_w.fix(75.38)  # J.mol-1.K-1
    model.tvc[model.q_w] = {0: 1.0, 100 / tau: ti_controls[2], (100 + ti_controls[1]) / tau: 1.00}
    model.hf_w.fix(-285830)  # J/mol
    model.tvc[model.Tw_in] = {0: 273.15, 1: 273.15}

    model.m_w.fix(18)  # g/mol
    model.rho_w.fix(1000)  # g/L

    """ simulating """
    simulator = PyosensSimulator(model, package="casadi")
    num1 = sum(sampling_times < 100).item()
    num2 = sum(sampling_times >= 100).item()
    if num1 == 0:
        num1 = (num2-1)*100//(int(round(sampling_times[-1]-100)))
    numpoints = num1+num2

    simulator.detect_sensitivity_analyis = True
    try:
        t, profile = simulator.simulate(
            integrator='idas',
            varying_inputs=model.tvc,
            numpoints=numpoints,
        )
    except RuntimeError:
        print("Model Parameters:")
        print(po.value(model.theta_0))
        print(po.value(model.theta_1))
        print(po.value(model.alpha_a))
        print(po.value(model.nu))

        print("Time-invariant Controls:")
        print(ti_controls)
        print("Sampling Time Candidates:")
        print(sampling_times)
        raise RuntimeError
    if False:
        plt.plot(t, profile)
        plt.show()
    diffvars = simulator.get_variable_order()
    # for var in diffvars:
    #     print(var.args[1])
    if simulator.do_sensitivity_analysis:
        resp, sens = profile
        if len(t) > numpoints:
            tswitch = t.searchsorted((100 + ti_controls[1]) / tau)
            resp2 = np.vstack((resp[0:tswitch], resp[tswitch+1:]))
            sens2 = np.vstack((sens[0:tswitch], sens[tswitch + 1:]))
            resp3 = resp2[num1:, [0, 1, 4, 5, 3]]
            sens3 = sens2[num1:, [0, 1, 4, 5, 3], :]
            profile = resp3, sens3
        else:
            resp3 = resp[num1:, [0, 1, 4, 5, 3]]
            sens3 = sens[num1:, [0, 1, 4, 5, 3]]
            profile = resp3, sens3
    else:
        resp = profile
        if len(t) > numpoints:
            tswitch = t.searchsorted((100 + ti_controls[1]) / tau)
            resp2 = np.vstack((resp[0:tswitch], resp[tswitch+1:]))
            resp3 = resp2[num1:, [0, 1, 4, 5, 3]]
            profile = resp3
        else:
            resp3 = resp[num1:, [0, 1, 4, 5, 3]]
            profile = resp3
    simulator.initialize_model()
    discretizer = po.TransformationFactory("dae.collocation")
    discretizer.apply_to(model, nfe=len(sampling_times), ncp=3, scheme="LAGRANGE-RADAU")

    """" extracting results and returning it in appropriate format """
    normalized_sampling_times = sampling_times / tau
    ca = np.array([model.ca[t].value for t in normalized_sampling_times if t >= 100 / tau])
    cb = np.array([model.cb[t].value for t in normalized_sampling_times if t >= 100 / tau])
    volume = np.array([model.v[t].value for t in normalized_sampling_times if t >= 100 / tau])
    reactor_temp = np.array([model.T[t].value for t in normalized_sampling_times if t >= 100 / tau])
    jacket_temp = np.array([model.Tj[t].value for t in normalized_sampling_times if t >= 100 / tau])

    # return np.array([ca, cb, reactor_temp, jacket_temp, volume]).T
    return profile

def simulate2_backup(ti_controls, sampling_times, model_parameters):
    tau = np.max(sampling_times)
    model = create_model(sampling_times)

    model.tau.fix(max(sampling_times))

    model.theta_0.fix(model_parameters[0])
    model.theta_1.fix(model_parameters[1])
    model.alpha_a.fix(model_parameters[2])
    model.alpha_b.fix(0)
    model.nu.fix(model_parameters[3])

    v0 = 2  # L
    model.v[0].fix(v0)  # L

    rho_c = 889  # g/L
    m_c = 72.1057  # g/mol
    cc0 = v0 * rho_c / m_c

    model.ca[0].fix(0.03)
    model.cb[0].fix(0)  # mol/L
    model.cc[0].fix(cc0)

    """ time-varying controls """
    model.tvc[model.q_in] = {0: 0.50, 100 / tau: ti_controls[0], (100+ti_controls[1]) / tau: 0.50}
    model.tvc[model.q_out] = {0: 0.50, 100 / tau: ti_controls[0], (100+ti_controls[1]) / tau: 0.50}
    model.tvc[model.ca_in] = {0: 0.5}
    model.tvc[model.cb_in] = {0: 0}
    model.tvc[model.cc_in] = {0: cc0}

    """ rm heat balance variables """
    model.T[0].fix(298.15)
    model.t_ref.fix(273.15)

    model.hf_a.fix(-80000)
    model.hf_b.fix(-180000)
    model.hf_c.fix(-123000)

    model.cp_a.fix(112.4)
    model.cp_b.fix(120)
    model.cp_c.fix(130)

    model.tvc[model.T_in] = {0: 273.15 + 25}

    model.U.fix(model_parameters[4])
    model.A.fix(5)

    """ jacket heat balance variables """
    model.Tj[0].fix(273.15)
    model.vj.fix(2)  # L
    model.cp_w.fix(75.38)  # J.mol-1.K-1
    model.tvc[model.q_w] = {0: ti_controls[2], }
    model.hf_w.fix(-285830)  # J/mol
    model.tvc[model.Tw_in] = {0: 273.15}

    model.m_w.fix(18)  # g/mol
    model.rho_w.fix(1000)  # g/L

    """ simulating """
    simulator = pod.Simulator(model, package="casadi")
    try:
        t, profile = simulator.simulate(
            integrator='idas',
            varying_inputs=model.tvc,
            numpoints=len(sampling_times)+1,
        )
    except RuntimeError:
        print("Model Parameters:")
        print(po.value(model.theta_0))
        print(po.value(model.theta_1))
        print(po.value(model.alpha_a))
        print(po.value(model.nu))

        print("Time-invariant Controls:")
        print(ti_controls)
        print("Sampling Time Candidates:")
        print(sampling_times)
        raise RuntimeError
    if False:
        plt.plot(t, profile)
        plt.show()
    simulator.initialize_model()
    discretizer = po.TransformationFactory("dae.collocation")
    discretizer.apply_to(model, nfe=len(sampling_times), ncp=3, scheme="LAGRANGE-RADAU")

    """" extracting results and returning it in appropriate format """
    normalized_sampling_times = sampling_times / tau
    ca = np.array([model.ca[t].value for t in normalized_sampling_times if t >= 100 / tau])
    cb = np.array([model.cb[t].value for t in normalized_sampling_times if t >= 100 / tau])
    volume = np.array([model.v[t].value for t in normalized_sampling_times if t >= 100 / tau])
    reactor_temp = np.array([model.T[t].value for t in normalized_sampling_times if t >= 100 / tau])
    jacket_temp = np.array([model.Tj[t].value for t in normalized_sampling_times if t >= 100 / tau])

    return np.array([ca, cb, reactor_temp, jacket_temp, volume]).T

def g_func(d, p):
    sampling_times = np.linspace(0, 200, 101)
    g_mat = []
    for cont in d:
        mono_c_g = []
        for param in p:
            res = simulate2(cont, sampling_times, param)

            # cb_ub = 0.260
            cb_lb = 0.240
            t_ub = 30 + 273.15

            conc_check1 = res[:, 1] - cb_lb
            # conc_check2 = cb_ub - res[:, 1]
            temp_check = t_ub - res[:, 2]

            # mono_c_g.append(np.array([conc_check1, conc_check2, temp_check]).T)
            mono_c_g.append(np.array([conc_check1, temp_check]).T)
        g_mat.append(mono_c_g)
    g_mat = np.asarray(g_mat)
    g_mat = np.min(g_mat, axis=2)

    return g_mat


if __name__ == '__main__':
    # check g_func
    if False:
        pre_exp_constant = 1.2e17  # in 1/min
        activ_energy = 100000  # in J/mol
        theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
        theta_1 = activ_energy / (8.314159 * 273.15)
        mp = np.array([
            theta_0,
            theta_1,
            1,  # alpha_a
            1.5,  # nu
            -80000,  # hf_a in J/mol
            -180000,  # hf_b in J/mol
            -123000,  # hf_c in J/mol
            100,  # U in W.m-2.K-1
        ])
        mp_ub = mp * 0.95
        mp_lb = mp * 1.05
        # mp_ub[7] = 100
        # mp_lb[7] = 10
        spt = np.linspace(0, 150, 101)

        np.random.seed(123)
        n_scr = 5
        mp = np.random.uniform(
            low=mp_ub,
            high=mp_lb,
            size=(n_scr, mp.shape[0])
        )
        g_func(
            d=np.array([
                [0.20, 10],
                [0.30, 10],
                [0.30, 15],
            ]),
            p=mp,
        )
    # check simulate2
    if False:
        pre_exp_constant = 1.2e17  # in 1/min
        activ_energy = 100000  # in J/mol
        theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
        theta_1 = activ_energy / (8.314159 * 273.15)
        spt = np.linspace(0, 150, 101)
        res = simulate2(
            ti_controls=[0.20, 10],
            sampling_times=spt,
            model_parameters=np.array([
                theta_0,
                theta_1,
                1,  # alpha_a
                1.5,  # nu
                -80000,  # hf_a in J/mol
                -180000,  # hf_b in J/mol
                -123000,  # hf_c in J/mol
                100,  # U in W.m-2.K-1
            ])
        )
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot(
            spt[np.where(spt >= 100)],
            res[:, 0],
        )
        plt.show()
        res = simulate2(
            ti_controls=[1.0, 10],
            sampling_times=spt,
            model_parameters=np.array([
                theta_0,
                theta_1,
                1,  # alpha_a
                1.5,  # nu
                -80000,  # hf_a in J/mol
                -180000,  # hf_b in J/mol
                -123000,  # hf_c in J/mol
                100,  # U in W.m-2.K-1
            ])
        )
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot(
            spt[np.where(spt >= 100)],
            res[:, 0],
        )
        plt.show()

    """ Simulate Function Checks """
    # example fed-batch dynamics
    if False:
        tic = [
            0.01,  # initial A concentration mol/L
        ]
        tvc = [
            {0: 0, 10: 0.01, 20: 0},  # inlet flow rate
            {0: 0},  # q_out
            {0: 0},  # q_w
        ]
        spt = np.linspace(0, 30, 101)
    # fully-batch mode
    if False:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 0},  # inlet flow rate
            {0: 0},  # q_out
            {0: 0},  # q_w
        ]
        spt = np.linspace(0, 10, 101)
    # continuous mode at 0.35 L/min throughput, startup = 30 minutes, max temp = 70!
    if False:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 0.35},  # inlet flow rate
            {0: 0.35},  # q_out
            {0: 10},  # q_w
        ]
        spt = np.linspace(0, 60, 101)
    # continuous mode at 1 L/min throughput, unstable, max Temp = 80!
    if False:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 1.00},  # inlet flow rate
            {0: 1.00},  # q_out
            {0: 10},  # q_w
        ]
        spt = np.linspace(0, 10, 101)
    # continuous mode, okay startup = ~100 minutes, max Temp = 51
    if False:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 0.10},  # inlet flow rate
            {0: 0.10},  # q_out
            {0: 0.10},  # q_w
        ]
        spt = np.linspace(0, 150, 201)
    # continuous mode, no cooling water replenishment, suprisingly no explosion! Max temp = 54 because feed is continuously fed at T = 25 Celsius, quenches reaction mixture
    if False:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 0.10},  # inlet flow rate
            {0: 0.10},  # q_out
            {0: 0.00},  # q_w
        ]
        spt = np.linspace(0, 60*24*7, 201)
    # continuously feed only into batch, no cooling water replenishment, suprisingly no explosion! Because feed is continuously fed at T = 25 Celsius, quenches reaction mixture
    if False:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 0.1},  # inlet flow rate
            {0: 0},  # q_out
            {0: 0},  # q_w
        ]
        spt = np.linspace(0, 60*24*7, 201)
    # original continuous mode at 0.50 L/min of feed
    if True:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 0.50},  # q_in
            {0: 0.50},  # q_out
            {0: 1.00},  # q_w
        ]
        spt = np.linspace(0, 200, 201)
    # direct shift to 1.00 L/min of feed
    if False:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 0.50, 100: 1.00},  # q_in
            {0: 0.50, 100: 1.00},  # q_out
            {0: 1.1, 80: 1.2},  # q_w
        ]
        spt = np.linspace(0, 60 * 4, 201)
    # gradual shifting sequence to process 1.0 L/min
    if False:
        tic = [
            0.03,
        ]
        tvc = [
            {0: 0.50, 30: 0.60, 60: 0.70, 90: 0.80, 120: 0.90, 150: 1.00},  # q_in
            {0: 0.50, 30: 0.60, 60: 0.70, 90: 0.80, 120: 0.90, 150: 1.00},  # q_out
            {0: 1.1},  # q_w
        ]
        spt = np.linspace(0, 60*4, 201)

    # pre_exp_constant = 1.2e17  # in 1/min
    pre_exp_constant = 2.2e17  # in 1/min
    activ_energy = 100000  # in J/mol
    theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
    theta_1 = activ_energy / (8.314159 * 273.15)
    mp = np.array([
        theta_0,
        theta_1,
        1.0,  # alpha_a
        1.0,  # nu
        400,  # U in W.m-2.K-1
    ])

    response = simulate(tic, tvc, spt, mp)

    tic = [
        [0.5, 1, 1.00],
        [0.5, 1, 0.00],
    ]
    g_mat = g_func(
        d=tic,
        p=np.array([mp, mp]),
    )

    fig = plt.figure(figsize=(10, 4))
    axes = fig.add_subplot(121)
    axes.plot(spt, response[:, 0], label=r"$c_A$")
    axes.plot(spt, response[:, 1], label=r"$c_B$")
    # axes.plot(spt, response[:, 6], label=r"$c_C$")
    axes.set_xlabel("Time (minutes)")
    axes.set_ylabel("Molar Concentration (M)")
    axes.legend()
    # axes.axhline(
    #     y=0.260,
    #     xmin=0,
    #     xmax=1,
    #     c="tab:green",
    #     ls="--",
    # )
    axes.axhline(
        y=0.240,
        xmin=0,
        xmax=1,
        c="tab:green",
        ls="--",
    )

    axes2 = fig.add_subplot(122)
    axes2.plot(spt, response[:, 2] - 273.15, label=r"Reaction Mixture")
    axes2.plot(spt, response[:, 3] - 273.15, label=r"Jacket Water")
    axes2.set_xlabel("Time (minutes)")
    axes2.set_ylabel("Temperature (Celsius)")
    axes2.legend()
    axes2.axhline(
        y=35,
        xmin=0,
        xmax=1,
        c="tab:green",
        ls="--",
    )
    fig.tight_layout()

    fig.savefig("business_as_usual.png", dpi=360)

    fig = plt.figure(figsize=(12, 9))
    axes = fig.add_subplot(221)
    axes.plot(spt, response[:, 0], label=r"$c_A$")
    axes.plot(spt, response[:, 1], label=r"$c_B$")
    # axes.plot(spt, response[:, 6], label=r"$c_C$")
    axes.set_xlabel("Time (minutes)")
    axes.set_ylabel("Molar Concentration (M)")
    axes.legend()

    # axes.axhline(
    #     y=0.260,
    #     xmin=0,
    #     xmax=1,
    #     c="tab:green",
    #     ls="--",
    # )
    axes.axhline(
        y=0.240,
        xmin=0,
        xmax=1,
        c="tab:green",
        ls="--",
    )

    axes2 = fig.add_subplot(222)
    axes2.plot(spt, response[:, 2] - 273.15, label=r"Reaction Mixture")
    axes2.plot(spt, response[:, 3] - 273.15, label=r"Jacket Water")
    axes2.set_xlabel("Time (minutes)")
    axes2.set_ylabel("Temperature (Celsius)")
    axes2.legend()
    axes2.axhline(
        y=35,
        xmin=0,
        xmax=1,
        c="tab:green",
        ls="--",
    )

    axes3 = fig.add_subplot(223)
    axes3.plot(spt, response[:, 4], label=r"Reaction Mixture")
    axes3.set_xlabel("Time (minutes)")
    axes3.set_ylabel("Volume (L)")
    axes3.legend()

    axes4 = fig.add_subplot(224)
    axes4.plot(spt, response[:, 5], label=r"Reaction Rate")
    axes4.set_xlabel("Time (minutes)")
    axes4.set_ylabel("Rate (mol/minute)")
    axes4.legend()

    fig.tight_layout()

    plt.show()
