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


def z2s_transform(z, size_scale=50):
    z_final = z
    z_final -= 10.5
    z_length = 20 - 1
    z_final /= 0.50 * z_length
    z_final *= size_scale
    z_final += 1.05 * size_scale
    return z_final

marker = "o"

figsize = (8, 6)
alpha = 0.50
translucent_alpha = 0.05
fig1 = plt.figure(figsize=figsize)
axes1 = fig1.add_subplot(111)

axes1.set_xlim([0.45, 2.55])
axes1.set_xlabel("$q$ (L/min)")
axes1.set_ylim([-.1, 4.1])
axes1.set_ylabel("$q_{w}$ (L/min)")

d_opt_eff = np.zeros((group4_samples.shape[0]))

d_opt_eff[167] = 0.2010
d_opt_eff[189] = 0.1211
d_opt_eff[193] = 0.0160
d_opt_eff[214] = 0.1180
d_opt_eff[255] = 0.2473
d_opt_eff[324] = 0.2187
d_opt_eff[405] = 0.0778

x = group1_samples[:, 0]
y = group1_samples[:, 2]
z = group1_samples[:, 1]
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
z = group2_samples[:, 1]
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
z = group3_samples[:, 1]
z = z2s_transform(z)
axes1.scatter(
    x,
    y,
    s=z,
    c='orange',
    marker=marker,
    alpha=translucent_alpha,
    label=f'${alphas[1]:.2f} \leq \\alpha <${alphas[2]:.2f}'
)
x = group4_samples[:, 0]
y = group4_samples[:, 2]
z = group4_samples[:, 1]
z = z2s_transform(z)

rgba_colors = np.zeros((x.shape[0], 4))
rgba_colors[:, 0] = 1.0
rgba_colors[:, 3] = np.ones((x.shape[0])) * translucent_alpha
rgba_colors[np.where(d_opt_eff > 1e-2), 3] = alpha
axes1.scatter(
    x,
    y,
    s=z,
    c=rgba_colors,
    marker=marker,
    label=f'${alphas[2]:.2f} \leq \\alpha \leq$1.00',
)
x = group4_samples[:, 0]
y = group4_samples[:, 2]
z = group4_samples[:, 1]
z = z2s_transform(z)
axes1.scatter(
    x,
    y,
    s=d_opt_eff * 3000,
    alpha=1.0,
    facecolors="none",
    edgecolors="r",
    marker="H",
    label=f'Optimal Effort',
)

""" RESTRICTED LOCAL """
axes1.annotate(
    "Experiment 168",
    xy=(1.62460249, 0.238817),
    xytext=(1.00, 0.50),
    arrowprops={
        "facecolor": "black",
        "connectionstyle": "arc3,rad=0.1",
        "arrowstyle": "-",
    }
)
axes1.annotate(
    "Experiment 190",
    xy=(1.91112756, 2.56384002),
    xytext=(1.00, 2.75),
    arrowprops={
        "facecolor": "black",
        "connectionstyle": "arc3,rad=-0.05",
        "arrowstyle": "-",
    }
)
axes1.annotate(
    "Experiment 194",
    xy=(1.94014266, 0.11000949),
    xytext=(1.10, 2.25),
    arrowprops={
        "facecolor": "black",
        "connectionstyle": "arc3,rad=-0.3",
        "arrowstyle": "-",
    }
)
axes1.annotate(
    "Experiment 215",
    xy=(2.30984255, 0.08305171),
    xytext=(2.00, 1.50),
    arrowprops={
        "facecolor": "black",
        "connectionstyle": "arc3,rad=-0.1",
        "arrowstyle": "-",
    }
)
axes1.annotate(
    "Experiment 256",
    xy=(1.76043295, 0.19238125),
    xytext=(1.10, 1.75),
    arrowprops={
        "facecolor": "black",
        "connectionstyle": "arc3,rad=-0.3",
        "arrowstyle": "-",
    }
)
axes1.annotate(
    "Experiment 325",
    xy=(2.07492807, 0.15897416),
    xytext=(1.60, 2.10),
    arrowprops={
        "facecolor": "black",
        "connectionstyle": "arc3,rad=-0.2",
        "arrowstyle": "-",
    }
)
axes1.annotate(
    "Experiment 406",
    xy=(1.68371149, 0.45278905),
    xytext=(1.10, 1.25),
    arrowprops={
        "facecolor": "black",
        "connectionstyle": "arc3,rad=-0.2",
        "arrowstyle": "-",
    }
)
axes1.legend()
fig1.tight_layout()

fig1.savefig("restricted_local.png", dpi=360)
plt.show()
