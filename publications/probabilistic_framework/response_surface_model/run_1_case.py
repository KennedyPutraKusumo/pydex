from stage_1_sample_design_space import draw_samples, ModelA
from oed import design_restricted_experiment, design_experiment_given_samples
from matplotlib import pyplot as plt
from time import time
from os import getcwd
from pickle import load
from pydex.core.designer import Designer
from scipy.special import erf
import numpy as np
import csv


def run(case_id, point_schedule, n_samples_p):
    n_lp = point_schedule[-1][1]
    case_name = draw_samples(case_id, point_schedule, n_samples_p)
    designer, fig = design_restricted_experiment(case_id, point_schedule, n_samples_p)


def stitch_candidates(cases):
    total_safe_candidates = None
    for case in cases:
        n_lp, n_samples_p = case
        case_name = f"ring_safe_{n_lp}_lp_{n_samples_p}_scr"

        alpha = 0.85

        with open(getcwd() + "/raw_data/" + case_name + "/output.pkl", "rb") as file:
            ns_output = load(file)
        experimental_candidates = ns_output["solution"]["probabilistic_phase"]["samples"]
        safe_candidates = np.asarray(experimental_candidates["coordinates"])[
            np.where(np.asarray(experimental_candidates["phi"]) >= alpha)]
        if total_safe_candidates is None:
            total_safe_candidates = safe_candidates
        else:
            total_safe_candidates = np.append(total_safe_candidates, safe_candidates, axis=0)
    return total_safe_candidates


def run_stitch(cases):
    total_safe_candidates = stitch_candidates(cases)

    n_lp = total_safe_candidates.shape[0]
    n_samples_p = cases[0, 1]
    case_name = f"ring_safe_{n_lp}_lp_{n_samples_p}_scr"
    designer, fig = design_experiment_given_samples(total_safe_candidates)

    return designer, fig


def vary_n_scr_manual_filter():
    cases = np.array([
        [1000, 100],
        [1000, 1000],
        [1000, 2000],
        # [1000, 4000],
        # [1000, 6000],
        # [1000, 8000],
    ])
    total_lp = stitch_candidates(cases)

    alpha = 0.85

    for case in cases:
        n_samples_p = case[1]

        np.random.seed(1989)
        p_best = [2, 1, 1, 1, 2, 2]
        p_sdev = 0.05 * np.identity(6)
        p_samples = np.random.multivariate_normal(p_best, p_sdev, n_samples_p)
        model = ModelA()
        g = model.g(total_lp, p_samples)
        feasibility = np.all(g >= 0, axis=2)
        p_feas_of_lp = np.sum(feasibility, axis=1)
        samples = total_lp[p_feas_of_lp >= alpha*100]

        n_lp = samples.shape[0]
        n_samples_p = case[1]
        case_name = f"ring_safe_{n_lp}_lp_{n_samples_p}_scr"
        designer, fig = design_experiment_given_samples(samples, case_name=case_name)


start = time()

cases = np.array([
    [125, 1000],
    [250, 1000],
    [500, 1000],
    [1000, 1000],
    [2000, 1000],
])
designer, fig = run_stitch(cases)
ax = fig.get_axes()[0]
exp_candidates = ax.collections[0]
exp_candidates._sizes = [7]
ax.set_title(None)


""" Analytical Solution """

def Ft(x1, x2):
    return np.array([1, x1, x2, x1*x2, x1**2, x2**2])

def contours(theta, x1, x2):
    sigma_theta = 0.05 * np.eye(6)
    first = erf((3-Ft(x1, x2) @ theta)/(np.sqrt(2*(Ft(x1, x2)[:, None].T @ sigma_theta @ Ft(x1, x2)[:, None])[0, 0])))
    second = erf((1.85-Ft(x1, x2) @ theta)/(np.sqrt(2*(Ft(x1, x2)[:, None].T @ sigma_theta @ Ft(x1, x2)[:, None])[0, 0])))

    return first - second

if True:
    reso = 401
    theta = [2, 1, 1, 1, 2, 2]
    x = np.linspace(-1, 1, reso)
    y = np.linspace(-1, 1, reso)
    z = []
    for x_i in x:
        for y_i in y:
            z.append(contours(theta, x_i, y_i))
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    levels = [0, 2 * 0.85, 2 * 1.00]
    ax.tricontour(x, y, z, levels=levels, colors=["black", "blue"], zorder=0)

fig.tight_layout()
fig.savefig("ring_safe_candidates_with_analytic", dpi=360)
finish = time() - start
print(f"Single run took {finish} wallclock seconds.")

plt.show()
