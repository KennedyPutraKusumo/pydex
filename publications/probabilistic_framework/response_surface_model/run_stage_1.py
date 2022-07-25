from stage_1_sample_design_space import draw_samples, ModelA
from oed import design_restricted_experiment, design_experiment_given_samples
from matplotlib import pyplot as plt
from time import time
from os import getcwd
from pickle import load
from pydex.core.designer import Designer
import numpy as np
import csv


def run(case_id, point_schedule, n_samples_p):
    n_lp = point_schedule[-1][1]
    case_name = draw_samples(case_id, point_schedule, n_samples_p)
    designer, fig = design_restricted_experiment(case_id, point_schedule, n_samples_p)
    unstitched_writer.writerow([case_name, n_lp, n_samples_p, designer._criterion_value])


def stitch_candidates(cases):
    total_safe_candidates = None
    for case in cases:
        n_lp, n_samples_p = case
        case_name = f"ring_safe_{n_lp}_lp_{n_samples_p}_scr"

        alpha = 0.85

        with open(getcwd() + "/" + case_name + "/output.pkl", "rb") as file:
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
    stitched_writer.writerow([case_name, n_lp, n_samples_p, designer._criterion_value])

    return designer, fig


def vary_n_scr_manual_filter():
    cases = np.array([
        [1000, 100],
        [1000, 200],
        [1000, 400],
        # [1000, 1000],
        # [1000, 2000],
        # [1000, 4000],
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
        stitched_n_scr_writer.writerow([case_name, n_lp, n_samples_p, designer._criterion_value])


run_without_stitch = False
run_cases_with_stich = True
run_stitch_vary_n_scr = False

start = time()
""" Cases without Stitching """
if run_without_stitch:
    f = open("safe_oed_ring_results.csv", "w", newline="")
    unstitched_writer = csv.writer(
        f,
        delimiter=",",
    )
    unstitched_writer.writerow(
        ["Case Name", "Number of Live Points", "Number of Scenarios", "Criterion Value"])

# case 1: 1000 live points, 100 scenarios
if run_without_stitch:
    case_id = 1
    point_schedule = [
        (.00, 100, 30),
        (.01, 250, 100),
        (.50, 500, 100),
        (.80, 1000, 100),
    ]
    n_samples_p = 100
    run(case_id, point_schedule, n_samples_p)

# case 2: 2000 live points, 100 scenarios
if run_without_stitch:
    case_id = 2
    point_schedule = [
        (.0, 200, 60),
        (.01, 500, 200),
        (.5, 1000, 200),
        (.8, 2000, 200),
    ]
    n_samples_p = 100
    run(case_id, point_schedule, n_samples_p)

# case 3: 2000 live points, 200 scenarios
if run_without_stitch:
    case_id = 3
    point_schedule = [
        (.0, 200, 60),
        (.01, 500, 200),
        (.5, 1000, 200),
        (.8, 2000, 200),
    ]
    n_samples_p = 200
    run(case_id, point_schedule, n_samples_p)

# case 4: 1000 live points, 1000 scenarios
if run_without_stitch:
    case_id = 4
    point_schedule = [
        (.0, 100, 30),
        (.01, 250, 100),
        (.5, 500, 100),
        (.8, 1000, 100),
    ]
    n_samples_p = 1000
    run(case_id, point_schedule, n_samples_p)

# case 5: 2000 live points, 1000 scenarios
if run_without_stitch:
    case_id = 5
    point_schedule = [
        (.0, 200, 60),
        (.01, 500, 200),
        (.5, 1000, 200),
        (.8, 2000, 200),
    ]
    n_samples_p = 1000
    run(case_id, point_schedule, n_samples_p)

# case 6: 4000 live points, 1000 scenarios
if run_without_stitch:
    case_id = 6
    point_schedule = [
        (.0, 400, 120),
        (.01, 1000, 400),
        (.5, 2000, 400),
        (.8, 4000, 400),
    ]
    n_samples_p = 1000
    run(case_id, point_schedule, n_samples_p)

# case 7: 1000 live points, 2000 scenarios
if run_without_stitch:
    case_id = 7
    point_schedule = [
        (.0, 100, 30),
        (.01, 250, 100),
        (.5, 500, 100),
        (.8, 1000, 100),
    ]
    n_samples_p = 2000
    run(case_id, point_schedule, n_samples_p)

# case 8: 1000 live points, 4000 scenarios
if run_without_stitch:
    case_id = 8
    point_schedule = [
        (.0, 100, 30),
        (.01, 250, 100),
        (.5, 500, 100),
        (.8, 1000, 100),
    ]
    n_samples_p = 4000
    run(case_id, point_schedule, n_samples_p)

# case 9: 1000 live points, 6000 scenarios
if run_without_stitch:
    case_id = 9
    point_schedule = [
        (.0, 100, 30),
        (.01, 250, 100),
        (.5, 500, 100),
        (.8, 1000, 100),
    ]
    n_samples_p = 6000
    run(case_id, point_schedule, n_samples_p)

# case 10: 1000 live points, 8000 scenarios
if run_without_stitch:
    case_id = 10
    point_schedule = [
        (.0, 100, 30),
        (.01, 250, 100),
        (.5, 500, 100),
        (.8, 1000, 100),
    ]
    n_samples_p = 8000
    run(case_id, point_schedule, n_samples_p)

# case 11: 4000 live points, 100 scenarios
if run_without_stitch:
    case_id = 11
    point_schedule = [
        (.0, 400, 120),
        (.01, 1000, 400),
        (.5, 2000, 400),
        (.8, 4000, 400),
    ]
    n_samples_p = 100
    run(case_id, point_schedule, n_samples_p)

# case 12: 8000 live points, 100 scenarios
if run_without_stitch:
    case_id = 12
    point_schedule = [
        (.0, 800, 240),
        (.01, 2000, 800),
        (.5, 4000, 800),
        (.8, 8000, 800),
    ]
    n_samples_p = 100
    run(case_id, point_schedule, n_samples_p)

# case 13: 500 live points, 1000 scenarios
if run_without_stitch:
    case_id = 13
    point_schedule = [
        (.00, 50, 15),
        (.01, 125, 50),
        (.50, 250, 50),
        (.80, 500, 50),
    ]
    n_samples_p = 1000
    run(case_id, point_schedule, n_samples_p)

# case 14: 250 live points, 1000 scenarios
if run_without_stitch:
    case_id = 14
    point_schedule = [
        (.00, 25, 7),
        (.01, 60, 25),
        (.50, 125, 25),
        (.80, 250, 25),
    ]
    n_samples_p = 1000
    run(case_id, point_schedule, n_samples_p)

# case 15: 125 live points, 1000 scenarios
if run_without_stitch:
    case_id = 15
    point_schedule = [
        (.00, 15, 7),
        (.01, 30, 12),
        (.50, 60, 12),
        (.80, 125, 12),
    ]
    n_samples_p = 1000
    run(case_id, point_schedule, n_samples_p)

# case 16: 1000 live points, 200 scenarios
if run_without_stitch:
    case_id = 16
    point_schedule = [
        (.0, 100, 30),
        (.01, 250, 100),
        (.5, 500, 100),
        (.8, 1000, 100),
    ]
    n_samples_p = 200
    run(case_id, point_schedule, n_samples_p)

# case 17: 1000 live points, 400 scenarios
if run_without_stitch:
    case_id = 17
    point_schedule = [
        (.0, 100, 30),
        (.01, 250, 100),
        (.5, 500, 100),
        (.8, 1000, 100),
    ]
    n_samples_p = 400
    run(case_id, point_schedule, n_samples_p)

# case 18: 1000 live points, 800 scenarios
if run_without_stitch:
    case_id = 18
    point_schedule = [
        (.0, 100, 30),
        (.01, 250, 100),
        (.5, 500, 100),
        (.8, 1000, 100),
    ]
    n_samples_p = 800
    run(case_id, point_schedule, n_samples_p)

""" Cases with Stitching """
if run_cases_with_stich:
    f = open("stitched_safe_oed_ring_results.csv", "w", newline="")
    stitched_writer = csv.writer(
        f,
        delimiter=",",
    )
    stitched_writer.writerow(
        ["Case Name", "Number of Live Points", "Number of Scenarios", "Criterion Value"])

if run_cases_with_stich:
    cases = np.array([
        [125, 1000],
    ])
    run_stitch(cases)

if run_cases_with_stich:
    cases = np.array([
        [125, 1000],
        [250, 1000],
    ])
    run_stitch(cases)

if run_cases_with_stich:
    cases = np.array([
        [125, 1000],
        [250, 1000],
        [500, 1000],
    ])
    run_stitch(cases)

if run_cases_with_stich:
    cases = np.array([
        [125, 1000],
        [250, 1000],
        [500, 1000],
        [1000, 1000],
    ])
    run_stitch(cases)

if run_cases_with_stich:
    cases = np.array([
        [125, 1000],
        [250, 1000],
        [500, 1000],
        [1000, 1000],
        [2000, 1000],
    ])
    run_stitch(cases)

if run_cases_with_stich:
    cases = np.array([
        [125, 1000],
        [250, 1000],
        [500, 1000],
        [1000, 1000],
        [2000, 1000],
        [4000, 1000],
    ])
    run_stitch(cases)

""" Stitched and Varying n_scr """
if run_stitch_vary_n_scr:
    f = open("stitched_vary_n_scr_results.csv", "w", newline="")
    stitched_n_scr_writer = csv.writer(
        f,
        delimiter=",",
    )
    stitched_n_scr_writer.writerow(
        ["Case Name", "Number of Live Points", "Number of Scenarios", "Criterion Value"])
    vary_n_scr_manual_filter()

finish = time() - start
print(f"All run cases took {finish} wallclock seconds.")
f.close()
plt.show()
