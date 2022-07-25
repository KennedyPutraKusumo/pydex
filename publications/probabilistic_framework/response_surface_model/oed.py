from pydex.core.designer import Designer
from pickle import load
from os import getcwd
import numpy as np


def simulate_sens(ti_controls, model_parameters):
    inner_designer = Designer()
    return_sensitivities = inner_designer.detect_sensitivity_analysis_function()
    res = np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        # interaction term
        model_parameters[3] * ti_controls[0] * ti_controls[1] +
        # squared terms
        model_parameters[4] * ti_controls[0] ** 2 +
        model_parameters[5] * ti_controls[1] ** 2
    ])
    if return_sensitivities:
        sens = np.array([
            [
                [
                    1,
                    ti_controls[0],
                    ti_controls[1],
                    ti_controls[0] * ti_controls[1],
                    ti_controls[0] ** 2,
                    ti_controls[1] ** 2],
            ]
        ])
        return res, sens
    else:
        return res


def design_experiment(case_name, title=False):
    designer = Designer()
    designer.simulate = simulate_sens

    alpha = 0.85

    with open(getcwd() + "/raw_data/" + case_name + "/output.pkl", "rb") as file:
        ns_output = load(file)
    experimental_candidates = ns_output["solution"]["probabilistic_phase"]["samples"]
    safe_candidates = np.asarray(experimental_candidates["coordinates"])[
        np.where(np.asarray(experimental_candidates["phi"]) >= alpha)]
    designer.ti_controls_candidates = safe_candidates

    designer.model_parameters = np.ones(6)
    designer.use_finite_difference = False
    designer.initialize(verbose=2)

    criterion = designer.d_opt_criterion
    designer.design_experiment(criterion, write=False)
    designer.print_optimal_candidates()

    designer.ti_controls_names = [r"$x_1$", r"$x_2$"]
    fig = designer.plot_optimal_controls(non_opt_candidates=True)
    ax = fig.axes[0]
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_yticks([-1, -.5, 0, .5, 1])
    if title:
        ax.set_title(case_name)
    fig.tight_layout()
    fig.savefig(f"{case_name}.png", dpi=360)

    n_exps = [6, 7, 8, 9, 10, 12, 14, 20, 30, 40, 50]
    for n_exp in n_exps:
        designer.apportion(n_exp)
    return designer, fig


def design_restricted_experiment(case_id, point_schedule, n_samples_p):
    n_lp = point_schedule[-1][1]
    case_name = f"ring_safe_{n_lp}_lp_{n_samples_p}_scr"
    designer, fig = design_experiment(case_name)
    return designer, fig

def design_experiment_given_samples(samples, case_name=None, title=False):
    designer = Designer()
    designer.simulate = simulate_sens

    n_lp = samples.shape[0]
    designer.ti_controls_candidates = samples

    designer.model_parameters = np.ones(6)
    designer.use_finite_difference = False
    designer.initialize(verbose=2)

    criterion = designer.d_opt_criterion
    designer.design_experiment(criterion, write=False)
    designer.print_optimal_candidates()

    designer.ti_controls_names = [r"$x_1$", r"$x_2$"]
    fig = designer.plot_optimal_controls(non_opt_candidates=True)
    ax = fig.axes[0]
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_yticks([-1, -.5, 0, .5, 1])
    if title:
        if case_name is None:
            ax.set_title(f"stitched_total_{n_lp}_lp")
        else:
            ax.set_title(case_name)
    fig.tight_layout()
    if case_name is None:
        fig.savefig(f"stitched_total_{n_lp}_lp.png", dpi=360)
    else:
        fig.savefig(f"{case_name}.png", dpi=360)

    n_exps = [6, 7, 8, 9, 10, 12, 14, 20, 30, 40, 50]
    for n_exp in n_exps:
        designer.apportion(n_exp)
    return designer, fig
