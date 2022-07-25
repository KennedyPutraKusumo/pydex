from model import simulate_one_tic, simulate_tvc_feed, simulate_tvc
from model_order import simulate_order
from matplotlib import pyplot as plt
from deus import DEUS
from time import time
import numpy as np
import os
import pickle
import datetime

tvc_feed = True
fix_switch = True
isothermal = False
rxn_order = True

def g_matrix(d, p):
    spt = np.linspace(0, 1, 21)
    g_mat = []
    for ind_d in d:
        g_d = []
        for ind_p in p:
            if tvc_feed:
                if fix_switch:
                    tic = [
                        ind_d[0]
                    ]
                    tvc = [{
                        0: ind_d[1],
                        0.25: ind_d[2],
                        0.50: ind_d[3],
                        0.75: ind_d[4]
                    }]
                else:
                    tic = [
                        ind_d[0],
                        ind_d[1],
                        ind_d[2],
                        ind_d[3],
                        ind_d[4],
                    ]
                    tvc = []
            elif isothermal:
                tic = [
                    ind_d[0],
                    ind_d[1],
                    ind_d[2],
                ]
                tvc = []
            mp = ind_p
            if tvc_feed:
                if rxn_order:
                    g = simulate_order(
                        ti_controls=tic,
                        tv_controls=tvc,
                        sampling_times=spt,
                        model_parameters=mp,
                        feasibility=True,
                    )
                else:
                    g = simulate_tvc_feed(
                        ti_controls=tic,
                        tv_controls=tvc,
                        sampling_times=spt,
                        model_parameters=mp,
                        feasibility=True,
                    )
            elif isothermal:
                if rxn_order:
                    g = simulate_order(
                        ti_controls=tic,
                        tv_controls=tvc,
                        sampling_times=spt,
                        model_parameters=mp,
                        feasibility=True,
                    )
                else:
                    g = simulate_one_tic(
                        ti_controls=tic,
                        tv_controls=tvc,
                        sampling_times=spt,
                        model_parameters=mp,
                        feasibility=True,
                    )
            g_d.append(g)
        g_mat.append(g_d)
    g_mat = np.array(g_mat)
    return g_mat.reshape(d.shape[0], p.shape[0], 1)

def g_matrix_tvc(d, p):
    spt = np.linspace(0, 1, 21)
    g_mat = []
    for ind_d in d:
        g_d = []
        for ind_p in p:
            tic = [
                ind_d[3],
                ind_d[4],
            ]
            tvc = [{0.0: ind_d[0], 0.33: ind_d[1], 0.67: ind_d[2]}]
            mp = ind_p
            g = simulate_tvc(
                ti_controls=tic,
                tv_controls=tvc,
                sampling_times=spt,
                model_parameters=mp,
                feasibility=True,
            )
            g_d.append(g)
        g_mat.append(g_d)
    g_mat = np.array(g_mat)
    return g_mat.reshape(d.shape[0], p.shape[0], 1)

def draw_sample(case_id, point_schedule=None, n_samples_p=10, target_reliability=0.95):
    if point_schedule is None:
        point_schedule = [
                    (0.00,  30, 10),
                    (0.01,  50, 30),
                    (0.50, 100, 50),
                    (0.80, 150, 75),
        ]

    nominal_mp = [3.5e11 / 3600, 82500]
    p_sdev = np.diag(nominal_mp * np.array([0.30, 0.05]))

    np.random.seed(1)
    p_samples = np.random.multivariate_normal(nominal_mp, p_sdev, n_samples_p)
    if rxn_order:
        nominal_mp = nominal_mp + [1, 1]
        rng = np.random.default_rng(seed=123)
        p_samples = np.append(p_samples, rng.choice([1, 2], (n_samples_p, 2), p=[0.75, 0.25]), axis=1)
    p_samples = [{'c': p, 'w': 1.0 / n_samples_p} for p in p_samples]

    if tvc_feed:
        if fix_switch:
            design_variables = [
                    {"T": [65+273.15, 75+273.15]},
                    {"tv1": [0.0 / 3600, 0.1 / 3600]},
                    {"tv2": [0.0 / 3600, 0.1 / 3600]},
                    {"tv3": [0.0 / 3600, 0.1 / 3600]},
                    {"tv4": [0.0 / 3600, 0.1 / 3600]},
                ]
        else:
            design_variables = [
                    {"T": [65+273.15, 75+273.15]},
                    {"tv1": [0.03 / 3600, 0.1 / 3600]},
                    {"tv_tau1": [0.10, 0.50]},
                    {"tv2": [0.03 / 3600, 0.1 / 3600]},
                    {"tv_tau2": [0.0, 0.40]},
                ]
        constraints_func_ptr = g_matrix
    elif isothermal:
        design_variables = [
                {"T": [65+273.15, 75+273.15]},
                {"tv": [0.03 / 3600, 0.1 / 3600]},
                {"tv_tau": [0.10, 0.90]},
            ]
        constraints_func_ptr = g_matrix
    else:
        design_variables = [
                {"T1": [65+273.15, 75+273.15]},
                {"T2": [65+273.15, 75+273.15]},
                {"T3": [65+273.15, 75+273.15]},
                {"tv": [0.03 / 3600, 0.1 / 3600]},
                {"tv_tau": [0.10, 0.90]},
            ]
        constraints_func_ptr = g_matrix_tvc

    the_activity_form = {
        "activity_type": "ds",

        "activity_settings": {
            "case_name": case_id,
            "case_path": os.getcwd(),
            "resume": False,
            "save_period": 1
        },

        "problem": {
            "user_script_filename": "none",
            "constraints_func_name": "none",
            "parameters_best_estimate": nominal_mp,
            "parameters_samples": p_samples,
            "target_reliability": target_reliability,
            "design_variables": design_variables,
        },

        "solver": {
            "name": "ds-ns",
            "settings": {
                "score_evaluation": {
                    "method": "serial",
                    "constraints_func_ptr": constraints_func_ptr,
                    "store_constraints": False
                },
                "efp_evaluation": {
                    "method": "serial",
                    "constraints_func_ptr": constraints_func_ptr,
                    "store_constraints": False,
                    "acceleration": False,
                },
                "points_schedule": point_schedule,
                "stop_criteria": [
                    {"inside_fraction": 1.0}
                ]
            },
            "algorithms": {
                "sampling": {
                    "algorithm": "mc_sampling-ns_global",
                    "settings": {
                         "nlive": 10,  # This is overridden by points_schedule
                         "nreplacements": 5,  # This is overriden by points_schedule
                         "prng_seed": 1989,
                         "f0": 0.3,
                         "alpha": 0.3,
                         "stop_criteria": [
                             {"max_iterations": 100000}
                         ],
                         "debug_level": 0,
                         "monitor_performance": False
                     },
                    "algorithms": {
                        "replacement": {
                            "sampling": {
                                "algorithm": "suob-ellipsoid"
                            }
                        }
                    }
                }
            }
        }
    }

    the_duu = DEUS(the_activity_form)
    t0 = time()
    the_duu.solve()
    cpu_secs = time() - t0
    print('CPU seconds', cpu_secs)

    cs_path = the_activity_form["activity_settings"]["case_path"]
    cs_name = the_activity_form["activity_settings"]["case_name"]

    with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb') \
            as file:
        output = pickle.load(file)

    samples = output["solution"]["probabilistic_phase"]["samples"]
    inside_samples_coords = []
    outside_samples_coords = []
    for i, phi in enumerate(samples["phi"]):
        if phi >= target_reliability:
            inside_samples_coords.append(samples["coordinates"][i])
        else:
            outside_samples_coords.append(samples["coordinates"][i])
    inside_samples_coords = np.array(inside_samples_coords)
    outside_samples_coords = np.array(outside_samples_coords)

    return inside_samples_coords, outside_samples_coords, samples
if __name__ == '__main__':
    test_g_matrix = False
    if test_g_matrix:
        if tvc_feed:
            d = np.array([
                [60+273.15, 0.1/3600, 0.10, 0.03/3600, 0.50],
                [75+273.15, 0.1/3600, 0.20, 0.03/3600, 0.20],
                [70+273.15, 0.1/3600, 0.50, 0.03/3600, 0.10],
            ])
        elif isothermal:
            d = np.array([
                [60+273.15, 0.1/3600, 0.10],
                [75+273.15, 0.1/3600, 0.20],
                [70+273.15, 0.1/3600, 0.50],
            ])
        p = np.array([
            [2.5e11 / 3600, 81000],
            [3.0e11 / 3600, 82500],
            [3.5e11 / 3600, 80000],
            [4.5e11 / 3600, 84000],
        ])
        g_mat = g_matrix(d, p)
        print(g_mat)
    n_samples_p = 100
    point_schedule = [
        (0.00,  200,  70),
        (0.01,  340, 200),
        (0.50,  670, 340),
        (0.80, 1000, 470),
    ]
    if tvc_feed:
        if fix_switch:
            if rxn_order:
                case_id = f"{datetime.datetime.now().strftime('%y-%m-%d')}"\
                          + f"_{point_schedule[-1][1]}_lp_{n_samples_p}_n_scr_fix_switch_rxn_order"
            else:
                case_id = f"{datetime.datetime.now().strftime('%y-%m-%d')}"\
                          + f"_{point_schedule[-1][1]}_lp_{n_samples_p}_n_scr_fix_switch"
        else:
            if rxn_order:
                case_id = f"{datetime.datetime.now().strftime('%y-%m-%d')}"\
                          + f"_{point_schedule[-1][1]}_lp_{n_samples_p}_n_scr-tvc_feed_rxn_order"
            else:
                case_id = f"{datetime.datetime.now().strftime('%y-%m-%d')}"\
                          + f"_{point_schedule[-1][1]}_lp_{n_samples_p}_n_scr-tvc_feed"
    elif isothermal:
        if rxn_order:
            case_id = f"{datetime.datetime.now().strftime('%y-%m-%d')}"\
                      + f"_{point_schedule[-1][1]}_lp_{n_samples_p}_n_scr_rxn_order"
        else:
            case_id = f"{datetime.datetime.now().strftime('%y-%m-%d')}"\
                      + f"_{point_schedule[-1][1]}_lp_{n_samples_p}_n_scr"
    else:
        case_id = f"{datetime.datetime.now().strftime('%y-%m-%d')}"\
                  + f"_{point_schedule[-1][1]}_lp_{n_samples_p}_n_scr_tvc"
    feas, infeas, total = draw_sample(
        case_id,
        point_schedule=point_schedule,
        target_reliability=0.95,
        n_samples_p=n_samples_p,
    )
    if isothermal:
        fig1 = plt.figure()
        x = feas[:, 0]
        y = feas[:, 1]
        z = feas[:, 2]
        axes1 = fig1.add_subplot(111, projection="3d")
        axes1.scatter(x, y, z, s=10, c='r', alpha=0.5, label='inside')

        fig1.savefig(f"Probabilistic_DS_{point_schedule[-1][1]}_lp_{n_samples_p}_n_scr.png", dpi=360)

        # print(infeas)
        # x = infeas[:, 0]
        # y = infeas[:, 1]
        # axes1.scatter(x, y, s=10, c='b', alpha=0.5, label='outside')

        plt.show()
