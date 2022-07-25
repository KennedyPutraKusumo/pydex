from matplotlib import pyplot as plt
from cycler import cycler
import pickle
import numpy as np


def visualize(case_id, target_reliability=0.95):
    with open(case_id + "/output.pkl", "rb") as file:
        output = pickle.load(file)
    samples = output["solution"]["probabilistic_phase"]["samples"]
    feas = []
    infeas = []
    for i, phi in enumerate(samples["phi"]):
        if phi >= target_reliability:
            feas.append(samples["coordinates"][i])
        else:
            infeas.append(samples["coordinates"][i])
    feas = np.array(feas)
    infeas = np.array(infeas)

    fig1 = plt.figure()
    x = feas[:, 0]
    y = feas[:, 1]
    z = feas[:, 2]
    axes1 = fig1.add_subplot(111, projection="3d")
    axes1.scatter(x, y, z, s=10, c='r', alpha=0.5, label='inside')

    if infeas.size != 0:
        x = infeas[:, 0]
        y = infeas[:, 1]
        axes1.scatter(x, y, s=10, c='b', alpha=0.5, label='outside')

    axes1.legend()
    fig1.savefig(f"Probabilistic_DS_{case_id}.png", dpi=360)

    plt.show()

def visualize_groups(case_id, n_groups, target_reliability=0.95):
    with open(case_id + "/output.pkl", "rb") as file:
        output = pickle.load(file)
    samples = output["solution"]["probabilistic_phase"]["samples"]

    levels = np.linspace(0, target_reliability, n_groups)
    levels = np.append(levels, 1)
    levels = np.flip(levels, axis=0)
    custom_cycler_1 = cycler(color=[
        "red",
        "orange",
        "gold",
        "green",
        "blue",
        "purple",
    ])

    feas = {}
    for l in range(levels.size-1):
        feas[f"Group {l + 1}"] = []
    for i, phi in enumerate(samples["phi"]):
        for l in range(levels.size-1):
            if levels[l] >= phi >= levels[l+1]:
                feas[f"Group {l + 1}"].append(np.array(samples["coordinates"][i]))
    for l in range(levels.size-1):
        feas[f"Group {l + 1}"] = np.array(feas[f"Group {l + 1}"])

    fig1 = plt.figure()
    axes1 = fig1.add_subplot(111, projection="3d")
    axes1.set_prop_cycle(custom_cycler_1)
    axes1.set_xlabel("Reaction Temperature (K)")
    axes1.set_ylabel("Feedrate of B (L/s)")
    axes1.set_zlabel("Feeding Duration (Normalized Time)")
    for l in range(levels.size-1):
        if feas[f"Group {l+1}"].size != 0:
            x = feas[f"Group {l+1}"][:, 0]
            y = feas[f"Group {l+1}"][:, 1]
            z = feas[f"Group {l+1}"][:, 2]
            axes1.scatter(
                x,
                y,
                z,
                s=10,
                alpha=0.5,
                label=fr'${levels[l+1]:.2f} \leq \alpha \leq {levels[l]:.2f}$',
            )

    axes1.legend()
    fig1.tight_layout()
    fig1.savefig(f"Probabilistic_DS_{case_id}.png", dpi=360)

    plt.show()
    return

if __name__ == '__main__':
    case_id = "restricted_space_samples_DEUS"

    visualize_groups(
        case_id,
        n_groups=5,
        target_reliability=0.95,
    )
