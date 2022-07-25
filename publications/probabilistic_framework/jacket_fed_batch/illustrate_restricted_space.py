from matplotlib import pyplot as plt
import pickle
import numpy as np

fp = "restricted_space_samples"
with open(f'{fp}/output.pkl', 'rb') \
        as file:
    output = pickle.load(file)
alphas = [0.50, 0.80, 0.95]
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

print(f"There were {group4_samples.shape[0]} samples drawn from the target design space at {alphas[2]}% confidence.")


def z2s_transform(z, size_scale=70):
    z_final = z
    z_final -= 10.5
    z_length = 20 - 1
    z_final /= 0.50 * z_length
    z_final *= size_scale
    z_final += 1.05 * size_scale
    return z_final


figsize = (8, 6)
alpha = 0.50
translucent_alpha = 0.50
if False:
    fig1 = plt.figure(figsize=figsize)
    axes1 = fig1.add_subplot(111)

    axes1.set_xlim([0.45, 2.55])
    axes1.set_xlabel("$q$ (L/min)")
    axes1.set_ylim([1, 20])
    axes1.set_ylabel("Switch Duration (min)")

    marker = "o"

    x = group2_samples[:, 0]
    y = group2_samples[:, 1]
    z = np.copy(group2_samples[:, 2])
    z = z2s_transform(z)
    axes1.scatter(x, y, s=z, c='navy', marker=marker, alpha=0.5, label=f'${alphas[0]:.2f} \leq \\alpha <${alphas[1]:.2f}')

    x = group3_samples[:, 0]
    y = group3_samples[:, 1]
    z = np.copy(group3_samples[:, 2])
    z = z2s_transform(z)
    axes1.scatter(x, y, s=z, c='orange', marker=marker, alpha=0.5, label=f'${alphas[1]:.2f} \leq \\alpha <${alphas[2]:.2f}')

    x = group4_samples[:, 0]
    y = group4_samples[:, 1]
    z = np.copy(group4_samples[:, 2])
    z = z2s_transform(z)
    axes1.scatter(x, y, s=z, c='red', marker=marker, alpha=0.5, label=f'${alphas[2]:.2f} \leq \\alpha \leq$1.00')

    # d_opt_eff = pickle.load(open("d_opt_eff_2.pkl", "rb"))
    # d_opt_eff = np.sum(d_opt_eff, axis=1)
    # d_opt_eff[d_opt_eff < 1e-2] = 0
    # d_opt_eff = d_opt_eff / d_opt_eff.sum()

    x = group4_samples[:, 0]
    y = group4_samples[:, 1]
    z = np.copy(group4_samples[:, 2])
    z = z2s_transform(z)
    # axes1.scatter(
    #     x,
    #     y,
    #     s=d_opt_eff*2000,
    #     alpha=0.80,
    #     facecolors="none",
    #     edgecolors="r",
    #     marker="H",
    #     label=f'Optimal Effort',
    # )

    axes1.legend()
    fig1.tight_layout()

    fig1.savefig("d_opt_jfb_ds_opaque_tau_vs_qin.png", dpi=360)
if True:
    fig1 = plt.figure(figsize=figsize)
    axes1 = fig1.add_subplot(111)

    axes1.set_xlim([0.45, 2.55])
    axes1.set_xlabel("$q$ (L/min)")
    axes1.set_ylim([-0.10, 4.10])
    axes1.set_ylabel("$q_{w}$ (L/min)")

    marker = "o"

    """ Outside Samples """
    if True:
        x = group1_samples[:, 0]
        y = group1_samples[:, 2]
        z = np.copy(group1_samples[:, 1])
        z = z2s_transform(z)
        axes1.scatter(
            x,
            y,
            s=z,
            c='purple',
            marker=marker,
            alpha=translucent_alpha,
            label=f'$0.00 \leq \\alpha <${alphas[0]:.2f}'
        )

        x = group2_samples[:, 0]
        y = group2_samples[:, 2]
        z = np.copy(group2_samples[:, 1])
        z = z2s_transform(z)
        axes1.scatter(
            x,
            y,
            s=z,
            c='navy',
            marker=marker,
            alpha=translucent_alpha,
            label=f'${alphas[0]:.2f} \leq \\alpha <${alphas[1]:.2f}'
        )

        x = group3_samples[:, 0]
        y = group3_samples[:, 2]
        z = np.copy(group3_samples[:, 1])
        z = z2s_transform(z)
        axes1.scatter(x, y, s=z, c='orange', marker=marker, alpha=translucent_alpha,
                      label=f'${alphas[1]:.2f} \leq \\alpha <${alphas[2]:.2f}')

    x = group4_samples[:, 0]
    y = group4_samples[:, 2]
    z = np.copy(group4_samples[:, 1])
    z = z2s_transform(z)
    axes1.scatter(x, y, s=z, c='red', marker=marker, alpha=alpha,
                  label=f'${alphas[2]:.2f} \leq \\alpha \leq$1.00')
    if False:
        d_opt_eff = pickle.load(open("d_opt_eff_2.pkl", "rb"))
        d_opt_eff = np.sum(d_opt_eff, axis=1)
        d_opt_eff[d_opt_eff < 1e-2] = 0
        d_opt_eff = d_opt_eff / d_opt_eff.sum()

        x = group4_samples[:, 0]
        y = group4_samples[:, 2]
        z = np.copy(group4_samples[:, 1])
        z = z2s_transform(z)
        axes1.scatter(
            x,
            y,
            s=d_opt_eff * 2000,
            alpha=0.80,
            facecolors="none",
            edgecolors="r",
            marker="H",
            label=f'Optimal Effort',
        )

        axes1.legend()
        fig1.tight_layout()

        fig1.savefig("d_opt_jfb_ds_opaque_qw_vs_qin.png", dpi=360)
    if True:
        axes1.legend()
        fig1.tight_layout()

        fig1.savefig("jfb_ds_opaque_qw_vs_qin.png", dpi=360)
if False:
    fig1 = plt.figure(figsize=figsize)
    axes1 = fig1.add_subplot(111)

    axes1.set_xlim([0, 21])
    axes1.set_xlabel("Switch Duration (min)")
    axes1.set_ylim([-0.1, 2.10])
    axes1.set_ylabel("$q_{w}$ (L/min)")

    marker = "o"

    x = group2_samples[:, 1]
    y = group2_samples[:, 2]
    z = np.copy(group2_samples[:, 0])
    z = z2s_transform(z)
    axes1.scatter(x, y, s=z, c='navy', marker=marker, alpha=0.5,
                  label=f'${alphas[0]:.2f} \leq \\alpha <${alphas[1]:.2f}')

    x = group3_samples[:, 1]
    y = group3_samples[:, 2]
    z = np.copy(group3_samples[:, 0])
    z = z2s_transform(z)
    axes1.scatter(x, y, s=z, c='orange', marker=marker, alpha=0.5,
                  label=f'${alphas[1]:.2f} \leq \\alpha <${alphas[2]:.2f}')

    x = group4_samples[:, 1]
    y = group4_samples[:, 2]
    z = np.copy(group4_samples[:, 0])
    z = z2s_transform(z)
    axes1.scatter(x, y, s=z, c='red', marker=marker, alpha=0.5,
                  label=f'${alphas[2]:.2f} \leq \\alpha \leq$1.00')

    # d_opt_eff = pickle.load(open("d_opt_eff_2.pkl", "rb"))
    # d_opt_eff = np.sum(d_opt_eff, axis=1)
    # d_opt_eff[d_opt_eff < 1e-2] = 0
    # d_opt_eff = d_opt_eff / d_opt_eff.sum()
    #
    # x = group4_samples[:, 1]
    # y = group4_samples[:, 2]
    # z = np.copy(group4_samples[:, 0])
    # z = z2s_transform(z)
    # axes1.scatter(
    #     x,
    #     y,
    #     s=d_opt_eff * 2000,
    #     alpha=0.80,
    #     facecolors="none",
    #     edgecolors="r",
    #     marker="H",
    #     label=f'Optimal Effort',
    # )

    axes1.legend()
    fig1.tight_layout()

    fig1.savefig("d_opt_jfb_ds_opaque_qw_vs_tau.png", dpi=360)
plt.show()
