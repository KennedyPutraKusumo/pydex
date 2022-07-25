import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.stats import multivariate_normal
import time

from deus import DEUS


'''
Find a probabilistic design space of a given reliability value 'a'.
The model is the following:
    y = p1 + p2*d1 + p3*d2 + p4*d1*d2 + p5*d1^2 + p6*d2^2
    , where: 
    d1, d2 are design variables;
    p1, p2, p3, p4, p5, p6 are model parameters that has an uncertain value described by 
    a Gaussian distribution, N(m, sigma).
The constraints that must be fulfilled are the following:
    1.85 <= y <= 3  
'''


class ModelA:
    def __init__(self):
        pass

    def s(self, d, p):
        x1_final = []
        for ind_d in d:
            x1_inter = []
            for ind_p in p:
                d1, d2 = ind_d
                x1_inter.append(ind_p[0] + ind_p[1] * d1 + ind_p[2] * d2 + ind_p[3] * d1 * d2 + ind_p[4] * d1 ** 2 + ind_p[5] * d2 ** 2)
            x1_final.append(x1_inter)
        x1_final = np.asarray(x1_final)
        return x1_final

    def g(self, d, p):
        """
        :param d: a 2d array with N_live_points X n_d
        :param p: a 2d array with N_montecarlo_scr X n_p
        :return: a 3d array with N_live_points X N_montecarlo_scr X n_g

        n_d is the number of design variables
        n_p is the number of model parameters of the model
        n_g is the number of process/CQA constraints
        """
        s = self.s(d, p)
        g = []
        for ind_d, s_d in zip(d, s):
            g_d = []
            for ind_p, s_p in zip(p, s_d):
                g1 = s_p - 1.85
                g2 = 3.0 - s_p
                g_p = [g1, g2]
                g_d.append(g_p)
            g.append(g_d)
        g = np.asarray(g)
        return g

def draw_samples(case_id, point_schedule, n_samples_p, plot=False):

    print("".center(100, "="))
    print(f"case {case_id}: {point_schedule[-1][1]} live points, {n_samples_p} scenarios")
    print("".center(100, "="))

    case_name = f"ring_safe_{point_schedule[-1][1]}_lp_{n_samples_p}_scr"
    if os.path.isdir(os.getcwd()+"\\"+case_name):
        print("Directory with the same case name is detected, skipping sample drawing.")
        return case_name

    the_model = ModelA()

    p_best = [2, 1, 1, 1, 2, 2]
    p_sdev = 0.05 * np.identity(6)

    p_samples = np.random.multivariate_normal(p_best, p_sdev, n_samples_p)
    p_samples = [{'c': p, 'w': 1.0 / n_samples_p} for p in p_samples]
    np.random.seed(1)

    the_activity_form = {
        "activity_type": "ds",

        "activity_settings": {
            "case_name": case_name,
            "case_path": os.getcwd(),
            "resume": False,
            "save_period": 1
        },

        "problem": {
            "user_script_filename": "none",
            "constraints_func_name": "none",
            "parameters_best_estimate": p_best,
            "parameters_samples": p_samples,
            "target_reliability": 0.85,
            "design_variables": [
                {"d1": [-1.0, 1.0]},
                {"d2": [-1.0, 1.0]}
            ]
        },

        "solver": {
            "name": "ds-ns",
            "settings": {
                "score_evaluation": {
                    "method": "serial",
                    "constraints_func_ptr": the_model.g,
                    "store_constraints": False
                },
                "efp_evaluation": {
                    "method": "serial",
                    "constraints_func_ptr": the_model.g,
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
                         "f0": 0.1,
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
    t0 = time.time()
    the_duu.solve()
    cpu_secs = time.time() - t0
    print('CPU seconds', cpu_secs)

    cs_path = the_activity_form["activity_settings"]["case_path"]
    cs_name = the_activity_form["activity_settings"]["case_name"]

    with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb') \
            as file:
        output = pickle.load(file)

    samples = output["solution"]["probabilistic_phase"]["samples"]
    inside_samples_coords = np.empty((0, 2))
    outside_samples_coords = np.empty((0, 2))
    for i, phi in enumerate(samples["phi"]):
        if phi >= 0.85:
            inside_samples_coords = np.append(inside_samples_coords,
                                              [samples["coordinates"][i]], axis=0)
        else:
            outside_samples_coords = np.append(outside_samples_coords,
                                               [samples["coordinates"][i]], axis=0)

    if plot:
        fig1 = plt.figure()
        x = inside_samples_coords[:, 0]
        y = inside_samples_coords[:, 1]
        plt.scatter(x, y, s=10, c='r', alpha=1.0, label='inside')

        x = outside_samples_coords[:, 0]
        y = outside_samples_coords[:, 1]
        plt.scatter(x, y, s=10, c='b', alpha=0.5, label='outside')

        fig1.axes[0].set_title(case_name)

        fig2, ax = plt.subplots(1)
        x = [item["iteration"] for item in output["performance"]]
        y = [item["cpu_secs"]["proposals"]
             for item in output["performance"]]
        ax.plot(x, y, 'b-', label='proposals generation')

        x = [item["iteration"] for item in output["performance"]]
        y = [item["cpu_secs"]["phi_evals"]
             for item in output["performance"]]
        ax.plot(x, y, 'r-', label='phi evaluations')

        x = [item["iteration"] for item in output["performance"]]
        y = [item["cpu_secs"]["total"] for item in output["performance"]]
        ax.plot(x, y, 'g--', label='total')

        ax.set_ylabel('CPU seconds')
        ax.legend()

        fig3, ax = plt.subplots(1)
        x = [item["iteration"]
             for item in output["performance"]]
        y = [item["n_proposals"]
             for item in output["performance"]]
        line1 = ax.plot(x, y, 'k-', label='n proposals')

        x = [item["iteration"]
             for item in output["performance"]]
        y = [item["n_replacements"]
             for item in output["performance"]]
        line2 = ax.plot(x, y, 'g-', label='n replacements')

        x = [item["iteration"]
             for item in output["performance"]]
        y = [item["n_model_evals"]
             for item in output["performance"]]
        ax2 = ax.twinx()
        line3 = ax2.plot(x, y, 'b-', label='n model evals')

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc=0)

        plt.show()

    return case_name
