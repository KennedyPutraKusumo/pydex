from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np
import corner_mirror
import corner

def classify_samples(points, probabilities, intervals):
    classified_samples = {}
    for i, lvl in enumerate(intervals):
        classified_samples[i] = []
    classified_samples[len(intervals)] = []
    for p, prob in zip(points, probabilities):
        for i, lvl in enumerate(intervals):
            if i == 0:
                if 0 <= prob < lvl:
                    classified_samples[i].append([p, prob])
            else:
                if intervals[i-1] <= prob < lvl:
                    classified_samples[i].append([p, prob])
        if intervals[-1] <= prob <= 1:
            classified_samples[len(intervals)].append([p, prob])

    return classified_samples


def get_group_from_classified_samples(samples, group_no):
    grouped = np.array(samples[group_no])
    if len(grouped) != 0:
        return np.array([_ for _ in grouped[:, 0]])
    else:
        return None


def plot_results(points, probabilities, intervals, colors, axes_labels):
    grouped_samples = classify_samples(points, probabilities, intervals)
    fig1 = plt.figure(figsize=(13, 5))
    axes1 = fig1.add_subplot(121, projection="3d")
    axes2 = fig1.add_subplot(122, projection="3d")
    for group, data in grouped_samples.items():
        if group == 0:
            label = fr"$0.00 \leq \alpha \leq {intervals[group]:.2f}$"
        elif group == len(intervals):
            label = fr"${intervals[group - 1]} < \alpha \leq 1.00$"
        else:
            label = fr"${intervals[group - 1]:.2f} < \alpha \leq {intervals[group]:.2f}$"
        for i, p in enumerate(data):
            axes1.scatter(
                p[0][0],
                p[0][1],
                p[0][2],
                c=colors[group],
            )
            if i == 0:
                axes2.scatter(
                    p[0][0],
                    p[0][2],
                    p[0][3],
                    c=colors[group],
                    label=label,
                )
            else:
                axes2.scatter(
                    p[0][0],
                    p[0][2],
                    p[0][3],
                    c=colors[group],
                )
    axes1.set_xlabel(axes_labels[0][0])
    axes1.set_ylabel(axes_labels[0][1])
    axes1.set_zlabel(axes_labels[0][2])
    axes2.set_xlabel(axes_labels[1][0])
    axes2.set_ylabel(axes_labels[1][1])
    axes2.set_zlabel(axes_labels[1][2])

    axes2.legend()
    fig1.tight_layout()
    fig1.savefig("esterification_DS_4piece_qin.png", dpi=180)


if __name__ == '__main__':
    alpha = 0.05
    markersize = 4
    effort_size = 1000
    plot_efforts = True
    reorder_variables = True
    hide_diagonal_plots = True
    legend = True
    legend_size = 6
    local_design = True
    unrestricted = True
    offset_axes_limits = False

    c = ["magenta", "blue", "orange", "red"]
    prob_intervals = [0.70, 0.80, 0.95]
    if reorder_variables:
        labels = ["$u_1$ (L/s)", "$u_2$ (L/s)", "$u_3$ (L/s)", "$u_4$ (L/s)", "Temperature (K)", ]
    else:
        labels = ["Temperature (K)", "$u_1$ (L/s)", "$u_2$ (L/s)", "$u_3$ (L/s)", "$u_4$ (L/s)"]

    ax_lab = [
        ["Temperature (K)", "$q_{in}$ Period 1 (L/s)", "$q_{in}$ Period 2 (L/s)"],
        ["Temperature (K)", "$q_{in}$ Period 3 (L/s)", "$q_{in}$ Period 4 (L/s)"],
    ]

    with open("restricted_space_samples_DEUS/output.pkl", "rb") as file:
        data_set = pickle.load(file)
    samples = np.array(
        data_set['solution']['probabilistic_phase']['samples']['coordinates'])
    feas_probs = np.array(data_set["solution"]["probabilistic_phase"]['samples']['phi'])
    if reorder_variables:
        samples = samples[:, (1, 2, 3, 4, 0)]

    grouped_samples = classify_samples(
        samples,
        feas_probs,
        prob_intervals,
    )

    group3 = get_group_from_classified_samples(
        grouped_samples,
        3,
    )
    group2 = get_group_from_classified_samples(
        grouped_samples,
        2,
    )
    group1 = get_group_from_classified_samples(
        grouped_samples,
        1,
    )
    group0 = get_group_from_classified_samples(
        grouped_samples,
        0,
    )

    if offset_axes_limits:
        T_offset = 1
        u_offset = 0.3e-5
    else:
        T_offset = 0
        u_offset = 0
    if reorder_variables:
        plot_ranges = [
                (0 - u_offset, 2.8e-5 + u_offset),
                (0 - u_offset, 2.8e-5 + u_offset),
                (0 - u_offset, 2.8e-5 + u_offset),
                (0 - u_offset, 2.8e-5 + u_offset),
                (273.15 + 65 - T_offset, 273.15 + 75 + T_offset),
            ]
    else:
        plot_ranges = [
            (273.15 + 65 - T_offset, 273.15 + 75 + T_offset),
            (0 - u_offset, 2.8e-5 + u_offset),
            (0 - u_offset, 2.8e-5 + u_offset),
            (0 - u_offset, 2.8e-5 + u_offset),
            (0 - u_offset, 2.8e-5 + u_offset),
        ]

    fig = corner.corner(
        samples,
        plot_contours=False,
        plot_density=False,
        plot_datapoints=False,
        color="black",
        labels=labels,
        range=plot_ranges,
    )

    corner.overplot_points(
        fig,
        group3,
        alpha=alpha,
        color=c[3],
        ms=markersize,
    )

    corner.overplot_points(
        fig,
        group2,
        alpha=alpha,
        color=c[2],
        ms=markersize,
    )

    corner.overplot_points(
        fig,
        group1,
        alpha=alpha,
        color=c[1],
        ms=markersize,
    )

    corner.overplot_points(
        fig,
        group0,
        alpha=alpha,
        color=c[0],
        ms=markersize,
    )

    axes1 = fig.get_axes()
    axes1 = np.array(axes1).reshape((5, 5))

    if local_design:
        if unrestricted:
            with open("unrestricted_local_results/run_2/run_2_d_opt_criterion_oed_result.pkl", "rb") as file:
                pydex_result = pickle.load(file)
        else:
            with open("restricted_local_results/run_1/run_1_d_opt_criterion_oed_result.pkl", "rb") as file:
                pydex_result = pickle.load(file)

    else:
        with open("restricted_average_results/run_1/run_1_d_opt_criterion_oed_result.pkl", "rb") as file:
            pydex_result = pickle.load(file)
    pydex_samples = np.empty((pydex_result["ti_controls_candidates"].shape[0], 5))
    pydex_samples[:, 0] = pydex_result["ti_controls_candidates"][:, 0]
    pydex_samples[:, 1] = np.array(
        [t[0][0.0] for t in pydex_result["tv_controls_candidates"]])
    pydex_samples[:, 2] = np.array(
        [t[0][0.25] for t in pydex_result["tv_controls_candidates"]])
    pydex_samples[:, 3] = np.array(
        [t[0][0.5] for t in pydex_result["tv_controls_candidates"]])
    pydex_samples[:, 4] = np.array(
        [t[0][0.75] for t in pydex_result["tv_controls_candidates"]])

    if reorder_variables:
        pydex_samples = pydex_samples[:, (1, 2, 3, 4, 0)]

    if plot_efforts:
        effort_without_spt = np.sum(pydex_result["optimal_efforts"], axis=1)
        trimmed_efforts = np.where(effort_without_spt < 1e-3, 0, effort_without_spt)
        if unrestricted and local_design:
            corner_mirror.overplot_points(
                fig,
                pydex_samples,
                s=trimmed_efforts * effort_size,
                marker="H",
                facecolor="none",
                edgecolor="magenta",
            )
        else:
            corner_mirror.overplot_points(
                fig,
                pydex_samples,
                s=trimmed_efforts * effort_size,
                marker="H",
                facecolor="none",
                edgecolor="tab:red",
            )
        binary_efforts = np.sum(pydex_result["optimal_efforts"], axis=1)
        binary_efforts = np.where(binary_efforts > 1e-3, 1, 0)
        if unrestricted:
            corner_mirror.overplot_points(
                fig,
                pydex_samples,
                s=binary_efforts * markersize * 5,
                marker="o",
                color="magenta",
                alpha=0.50,
            )
        else:
            corner_mirror.overplot_points(
                fig,
                pydex_samples,
                s=binary_efforts * markersize * 5,
                marker="o",
                color="tab:red",
                alpha=0.50,
            )
        fig_title = "corner_DS_with_efforts"
    else:
        fig_title = "corner_DS"

    if reorder_variables:
        fig_title += "_reordered"

    axes = fig.get_axes()
    axes = np.array(axes).reshape((5, 5))

    if hide_diagonal_plots:
        for i in range(5):
            axes[i, i].set_visible(False)
        fig_title += "_no_diagonal"

    if legend:
        from matplotlib.lines import Line2D
        complete_intervals = [0.00]
        complete_intervals.extend(prob_intervals)
        interval_labels = []
        for i, interval in enumerate(complete_intervals):
            if i != len(complete_intervals) - 1:
                interval_labels.append(rf"${interval:.2f} \leq \alpha < {complete_intervals[i+1]:.2f}$")
            else:
                interval_labels.append(rf"${interval:.2f} \leq \alpha \leq 1.00$")

        custom_markers = [
            Line2D([0], [0], marker="o", alpha=0.4, color="w", markerfacecolor=colour, markeredgecolor=colour, label=label) for colour, label in zip(c, interval_labels)
        ]
        if unrestricted:
            axes[1, 0].legend(loc="upper left", handles=custom_markers, prop={"size": legend_size})
        else:
            axes[1, 0].legend(handles=custom_markers, prop={"size": legend_size})
        fig_title += "_with_legend"

    if local_design:
        if unrestricted:
            fig_title += "_unrestricted"
        fig_title += "_local"

    fig.tight_layout()
    fig.savefig(fig_title)
    plt.show()
