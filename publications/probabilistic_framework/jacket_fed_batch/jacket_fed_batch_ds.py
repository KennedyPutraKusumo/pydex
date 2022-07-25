import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
import pandas as pd


from deus import DEUS
from jacket_fed_batch import g_func

pre_exp_constant = 2.2e17  # in 1/min
activ_energy = 100000  # in J/mol
theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
theta_1 = activ_energy / (8.314159 * 273.15)
p_best = np.array([
    theta_0,
    theta_1,
    1.0,  # alpha_a
    1.0,  # nu
    400,  # U in W.m-2.K-1
])
mp_delta = 0.00
mp_ub = p_best * (1 - mp_delta)
mp_lb = p_best * (1 + mp_delta)

mp_ub[0] = p_best[0]
mp_lb[0] = 0.95 * p_best[0]

mp_ub[4] = 400
mp_lb[4] = 350

np.random.seed(123)
n_scr = 100
mp = np.random.uniform(
    low=mp_ub,
    high=mp_lb,
    size=(n_scr, p_best.shape[0])
)
p_samples = []
for p in mp:
    p_samples.append({
        "c": p,
        "w": 1/mp.shape[0],
    })

alpha = 0.95
case_name = "case1"
the_activity_form = {
    "activity_type": "ds",

    "activity_settings": {
        "case_name": f"jfb_ds_{alpha}_{case_name}_{n_scr}_run",
        "case_path": os.getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "user_script_filename": "none",
        "constraints_func_name": "none",
        "parameters_best_estimate": p_best.tolist(),
        "parameters_samples": p_samples,
        "target_reliability": alpha,
        "design_variables": [
            {"level": [0.5, 2.5]},
            {"switch_duration": [1, 20]},
            {"q_w": [0, 5]},
        ],
    },

    "solver": {
        "name": "ds-ns",
        "settings": {
            "score_evaluation": {
                "method": "serial",
                "constraints_func_ptr": g_func,
                "store_constraints": False,
            },
            "efp_evaluation": {
                "method": "serial",
                "constraints_func_ptr": g_func,
                "store_constraints": False,
                "acceleration": True,
            },
            "points_schedule": [
                (.0, 168, 21),
                (.01, 252, 50),
                (.5, 315, 50),
                (.8, 420, 105),
            ],
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
                     "alpha": 0.2,
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

print("")
the_duu = DEUS(the_activity_form)
t0 = time.time()
the_duu.solve()
cpu_secs = time.time() - t0
print('CPU seconds', cpu_secs)

fp = "jfb_ds_0.95_case1_100_run"
with open(f'{fp}/output.pkl', 'rb') \
        as file:
    output = pickle.load(file)
alphas = [0.10, 0.50, 0.80]
samples = output["solution"]["probabilistic_phase"]["samples"]
group1_samples = np.empty((0, 3))
group2_samples = np.empty((0, 3))
group3_samples = np.empty((0, 3))
group4_samples = np.empty((0, 3))
for coord, phi in zip(samples["coordinates"], samples["phi"]):
    if phi < alphas[0]:
        group1_samples = np.append(group1_samples, [coord], axis=0)
    elif phi < alphas[1]:
        group2_samples = np.append(group2_samples, [coord], axis=0)
    elif phi < alphas[2]:
        group3_samples = np.append(group3_samples, [coord], axis=0)
    else:
        group4_samples = np.append(group4_samples, [coord], axis=0)
