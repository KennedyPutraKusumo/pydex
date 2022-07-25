from model import simulate, simulate_one_tic
from model_order import simulate_order
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    alpha = 0.10

    k_nominal = 3.5e11 / 3600
    ea_nominal = 8.25e4

    # from uniform distribution
    if False:
        k_lb = 2.45e11 / 3600
        k_ub = 4.55e11 / 3600
        ea_lb = 7.84e4
        ea_ub = 8.66e4

        reso = 51j
        k_grid = np.mgrid[k_lb:k_ub:reso]
        ea_grid = np.mgrid[ea_lb:ea_ub:reso]

    # from normal distribution
    if True:
        n_scr = 100
        p_nominal = np.array([k_nominal, ea_nominal])
        p_sdev = np.diag(p_nominal * np.array([0.30, 0.05]))
        np.random.seed(1)
        p_samples = np.random.multivariate_normal(
            mean=p_nominal,
            cov=p_sdev,
            size=n_scr,
        )
        rngorder = np.random.default_rng(seed=123)
        p_samples = np.append(p_samples, rngorder.choice([1, 2], (n_scr, 2), p=[0.75, 0.25]), axis=1)
        k_grid = p_samples[:, 0]
        ea_grid = p_samples[:, 1]
        alpha_A_grid = p_samples[:, 2]
        alpha_B_grid = p_samples[:, 3]

    x_vals = []
    q_vals = []
    Tcf_vals = []
    fig1 = plt.figure()
    axes1 = fig1.add_subplot(221)
    axes1.set_xlabel("Normalized Time (hour/hour)")
    axes1.set_ylabel("Conversion of A (% - mol/mol)")
    axes2 = fig1.add_subplot(222)
    axes2.set_xlabel("Normalized Time (hour/hour)")
    axes2.set_ylabel("Power by Reaction (J/s)")
    axes3 = fig1.add_subplot(223)
    axes3.set_xlabel("Normalized Time (hour/hour)")
    axes3.set_ylabel("Conversion of A (% - mol/mol)")
    axes4 = fig1.add_subplot(224)
    axes4.set_xlabel("Normalized Time (hour/hour)")
    axes4.set_ylabel("Power by Reaction (J/s)")
    plot_mean = False
    separate_supports = True
    fig2 = plt.figure()
    if separate_supports:
        axesall = [fig2.add_subplot(321),
                 fig2.add_subplot(322),
                 fig2.add_subplot(323),
                 fig2.add_subplot(324),
                 fig2.add_subplot(325)]
    else:
        axes5 = fig2.add_subplot(111)
    separate_plots = False
    if separate_plots:
        # support 1 restricted average design
        if False:
            ti_controls = [
                2.5,  # N_A0 (mol)
                2.5,  # N_Bin (mol)
                0.32,  # V_A0 (L)
                0.23,  # V_Bin (L)
                30000,  # batch time (s)
                74.52 + 273.15,  # temperature (Kelvin)
            ]
            tv_controls = [
                {
                    0.00: 1.23e-5,
                    0.25: 0.91e-5,
                    0.50: 0.54e-5,
                    0.75: 0.66e-5,
                },
            ]
            name = "support_1"
        # support 2 restricted average design
        if False:
            ti_controls = [
                2.5,  # N_A0 (mol)
                2.5,  # N_Bin (mol)
                0.32,  # V_A0 (L)
                0.23,  # V_Bin (L)
                30000,  # batch time (s)
                348.14,  # temperature (Kelvin)
            ]
            tv_controls = [
                {
                    0.00: 1.19e-5,
                    0.25: 0.07e-5,
                    0.50: 0.19e-5,
                    0.75: 1.04e-5,
                },
            ]
            name = "support_2"
        # support 3 restricted average design
        if False:
            ti_controls = [
                2.5,  # N_A0 (mol)
                2.5,  # N_Bin (mol)
                0.32,  # V_A0 (L)
                0.23,  # V_Bin (L)
                30000,  # batch time (s)
                339.33,  # temperature (Kelvin)
            ]
            tv_controls = [
                {
                    0.00: 0.55e-5,
                    0.25: 0.07e-5,
                    0.50: 0.26e-5,
                    0.75: 0.69e-5,
                },
            ]
            name = "support_3"
        # support 4 restricted average design
        if False:
            ti_controls = [
                2.5,  # N_A0 (mol)
                2.5,  # N_Bin (mol)
                0.32,  # V_A0 (L)
                0.23,  # V_Bin (L)
                30000,  # batch time (s)
                338.34,  # temperature (Kelvin)
            ]
            tv_controls = [
                {
                    0.00: 1.16e-5,
                    0.25: 0.63e-5,
                    0.50: 0.27e-5,
                    0.75: 0.16e-5,
                },
            ]
            name = "support_4"
        # support 5 restricted average design
        if False:
            ti_controls = [
                2.5,  # N_A0 (mol)
                2.5,  # N_Bin (mol)
                0.32,  # V_A0 (L)
                0.23,  # V_Bin (L)
                30000,  # batch time (s)
                347.85,  # temperature (Kelvin)
            ]
            tv_controls = [
                {
                    0.00: 1.24e-5,
                    0.25: 0.19e-5,
                    0.50: 0.09e-5,
                    0.75: 0.17e-5,
                },
            ]
            name = "support_5"
    else:
        # support 1 restricted average design
        ti_controls_1 = [
            # 2.5,  # N_A0 (mol)
            # 2.5,  # N_Bin (mol)
            # 0.32,  # V_A0 (L)
            # 0.23,  # V_Bin (L)
            # 30000,  # batch time (s)
            74.52 + 273.15,  # temperature (Kelvin)
        ]
        tv_controls_1 = [
            {
                0.00: 1.23e-5,
                0.25: 0.91e-5,
                0.50: 0.54e-5,
                0.75: 0.66e-5,
            },
        ]
        name_1 = "Support 1"
        # support 2 restricted average design
        ti_controls_2 = [
            # 2.5,  # N_A0 (mol)
            # 2.5,  # N_Bin (mol)
            # 0.32,  # V_A0 (L)
            # 0.23,  # V_Bin (L)
            # 30000,  # batch time (s)
            348.14,  # temperature (Kelvin)
        ]
        tv_controls_2 = [
            {
                0.00: 1.19e-5,
                0.25: 0.07e-5,
                0.50: 0.19e-5,
                0.75: 1.04e-5,
            },
        ]
        name_2 = "Support 2"
        # support 3 restricted average design
        ti_controls_3 = [
            # 2.5,  # N_A0 (mol)
            # 2.5,  # N_Bin (mol)
            # 0.32,  # V_A0 (L)
            # 0.23,  # V_Bin (L)
            # 30000,  # batch time (s)
            339.33,  # temperature (Kelvin)
        ]
        tv_controls_3 = [
            {
                0.00: 0.55e-5,
                0.25: 0.07e-5,
                0.50: 0.26e-5,
                0.75: 0.69e-5,
            },
        ]
        name_3 = "Support 3"
        # support 4 restricted average design
        ti_controls_4 = [
            # 2.5,  # N_A0 (mol)
            # 2.5,  # N_Bin (mol)
            # 0.32,  # V_A0 (L)
            # 0.23,  # V_Bin (L)
            # 30000,  # batch time (s)
            338.34,  # temperature (Kelvin)
        ]
        tv_controls_4 = [
            {
                0.00: 1.16e-5,
                0.25: 0.63e-5,
                0.50: 0.27e-5,
                0.75: 0.16e-5,
            },
        ]
        name_4 = "Support 4"
        # support 5 restricted average design
        ti_controls_5 = [
            # 2.5,  # N_A0 (mol)
            # 2.5,  # N_Bin (mol)
            # 0.32,  # V_A0 (L)
            # 0.23,  # V_Bin (L)
            # 30000,  # batch time (s)
            347.85,  # temperature (Kelvin)
        ]
        tv_controls_5 = [
            {
                0.00: 1.24e-5,
                0.25: 0.19e-5,
                0.50: 0.09e-5,
                0.75: 0.17e-5,
            },
        ]
        name_5 = "Support 5"
        tic = [ti_controls_1, ti_controls_2, ti_controls_3, ti_controls_4, ti_controls_5]
        tvc = [tv_controls_1, tv_controls_2, tv_controls_3, tv_controls_4, tv_controls_5]
        names = [name_1, name_2, name_3, name_4, name_5]
        colours = ["tab:red", "gold", "tab:green", "tab:blue", "tab:purple"]
    sampling_times = np.linspace(0, 1, 101)
    if separate_plots:
        for k in k_grid:
            model_parameters = [
                k,
                ea_nominal,
            ]

            x, q, T_cf = simulate(
                ti_controls=ti_controls,
                tv_controls=tv_controls,
                sampling_times=sampling_times,
                model_parameters=model_parameters,
                plot=False,
            )
            x_vals.append(x)
            q_vals.append(q)
            Tcf_vals.append(T_cf)
            axes1.plot(
                sampling_times,
                x,
                c="tab:red",
                alpha=alpha,
            )
            axes2.plot(
                sampling_times,
                q,
                c="tab:red",
                alpha=alpha,
            )
        x_vals = np.array(x_vals)
        q_vals = np.array(q_vals)
        x_mean = np.mean(x_vals, axis=0)
        q_mean = np.mean(q_vals, axis=0)
        axes1.plot(
            sampling_times,
            x_mean,
            c="tab:red",
            alpha=1.0,
            ls="dashed",
        )
        axes2.plot(
            sampling_times,
            q_mean,
            c="tab:red",
            alpha=1.0,
            ls="dashed",
        )
        x_vals = []
        q_vals = []
        for ea in ea_grid:
            model_parameters = [
                k_nominal,
                ea,
            ]

            x, q, T_cf = simulate(
                ti_controls=ti_controls,
                tv_controls=tv_controls,
                sampling_times=sampling_times,
                model_parameters=model_parameters,
                plot=False,
            )
            x_vals.append(x)
            q_vals.append(q)
            axes3.plot(
                sampling_times,
                x,
                c="tab:red",
                alpha=alpha,
            )
            axes4.plot(
                sampling_times,
                q,
                c="tab:red",
                alpha=alpha,
            )
            axes5.plot(
                sampling_times * 500,
                T_cf,
                c="tab:red",
                alpha=alpha,
            )
            x_vals = np.array(x_vals)
            q_vals = np.array(q_vals)
            Tcf_vals = np.array(Tcf_vals)
            x_mean = np.mean(x_vals, axis=0)
            q_mean = np.mean(q_vals, axis=0)
            Tcf_mean = np.mean(Tcf_vals, axis=0)
            axes3.plot(
                sampling_times,
                x_mean,
                c="tab:red",
                alpha=1.0,
                ls="dashed",
            )
            axes4.plot(
                sampling_times,
                q_mean,
                c="tab:red",
                alpha=1.0,
                ls="dashed",
            )
            fig1.tight_layout()
            T_max = 405
            axes5.set_xlabel("Time (min)")
            axes5.set_ylabel("Temperature (K)")
            axes5.set_ylim([63 + 273.15, 133.85 + 273.15])
            axes5.legend()
            axes5.plot(
                sampling_times * 500,
                Tcf_mean,
                c="tab:red",
                alpha=1.0,
                ls="dashed",
            )
            axes5.axhline(
                y=T_max,
                xmin=0,
                xmax=1,
                c="tab:green",
                ls="dashed",
                label=f"Maximum Allowable Temperature ({T_max} Kelvin)",
            )

            fig2.tight_layout()
            fig2.savefig(f"{name}_feasibility.png", dpi=180)
    else:
        isup = 0
        for ti, tv, name, colour in zip(tic, tvc, names, colours):
            x_vals = []
            q_vals = []
            Tcf_vals = []
            if separate_supports:
                axes5 = axesall[isup]
            nviolate = 0

            for k, ea, alpha_A, alpha_B in zip(k_grid, ea_grid, alpha_A_grid, alpha_B_grid):
                model_parameters = [
                    k,
                    ea,
                    alpha_A,
                    alpha_B
                ]

                sol = simulate_order(
                    ti_controls=ti,
                    tv_controls=tv,
                    sampling_times=sampling_times,
                    model_parameters=model_parameters,
                    feasibility=False,
                    cooling_failure=True,
                )
                x = sol[:, 0]
                q = sol[:, 1]
                T_cf = sol[:, 2]
                T_max = 405
                gap = np.min(T_max - T_cf)
                if gap < 0.0:
                    nviolate = nviolate + 1
                    print(np.min(T_max - T_cf))
                x_vals.append(x)
                q_vals.append(q)
                Tcf_vals.append(T_cf)

                axes5.plot(
                    sampling_times * 500,
                    T_cf,
                    c=colour,
                    alpha=alpha,
                )
                axes5.set_ylabel("Temperature (K)", fontsize=8)
                axes5.set_xlabel("Time (min)", fontsize=8)

            if separate_supports:
                T_max = 405
                axes5.axhline(
                    y=T_max,
                    xmin=0,
                    xmax=1,
                    c="tab:green",
                    ls="dashed",
                    # label=r"$T^{\rm max}$ (405 Kelvin)",
                )
                axes5.set_title(name)
                # axes5.legend()
            if plot_mean:
                x_vals = np.array(x_vals)
                q_vals = np.array(q_vals)
                Tcf_vals = np.array(Tcf_vals)
                x_mean = np.mean(x_vals, axis=0)
                q_mean = np.mean(q_vals, axis=0)
                Tcf_mean = np.mean(Tcf_vals, axis=0)

                axes5.set_xlabel("Time (min)")
                axes5.set_ylabel(r"$T_{\rm cf}$ (K)")
                axes5.set_ylim([63 + 273.15, 133.85 + 273.15])
                axes5.plot(
                    sampling_times * 500,
                    Tcf_mean,
                    c=colour,
                    alpha=1.0,
                    label=name,
                )
            print("number of infeasible points", nviolate)
            isup = isup + 1

        if not separate_supports:
            T_max = 405
            axes5.axhline(
                y=T_max,
                xmin=0,
                xmax=1,
                c="tab:green",
                ls="dashed",
                label=r"$T^{\rm max}$ (405 Kelvin)",
            )
            axes5.legend()

        fig2.tight_layout()
        fig2.savefig(f"combined_feasibility_supports.png", dpi=180)

    plt.show()
