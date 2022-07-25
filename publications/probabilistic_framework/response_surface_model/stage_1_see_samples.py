import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.special import erf


cs_path = os.getcwd() + '/raw_data'
cs_name = "ring_safe_2000_lp_1000_scr"

with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb') as file:
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

""" DS Figure """
marker_size = 5
marker_alpha = 0.50
fig1 = plt.figure()
x = inside_samples_coords[:, 0]
y = inside_samples_coords[:, 1]
plt.scatter(x, y, s=marker_size, c='r', alpha=marker_alpha, label='inside')

x = outside_samples_coords[:, 0]
y = outside_samples_coords[:, 1]
plt.scatter(x, y, s=marker_size, c='b', alpha=marker_alpha, label='outside')

""" Deterministic DS Check """
if True:
    det_samples = np.asarray(output["solution"]["deterministic_phase"]["samples"]["coordinates"])
    x = det_samples[:, 0]
    y = det_samples[:, 1]
    plt.scatter(x, y, s=marker_size, c='g', alpha=marker_alpha, label='Deterministic')

fig1.axes[0].set_title(cs_name)
fig1.tight_layout()

""" Solution Times """
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
print(f"Total computational time: {np.sum(y):.2f} CPU seconds")

ax.set_ylabel('CPU seconds')
ax.legend()
fig2.tight_layout()

""" Number of Computations """
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
ax.set_ylabel("Number of Proposals or Replacements")

x = [item["iteration"]
     for item in output["performance"]]
y = [item["n_model_evals"]
     for item in output["performance"]]
ax2 = ax.twinx()
line3 = ax2.plot(x, y, 'b-', label='n model evals')
ax2.set_ylabel("Model Evaluations")

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc=0)
fig3.tight_layout()

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
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.tricontour(x, y, z, levels=levels, colors=["black", "blue"])
    x = inside_samples_coords[:, 0]
    y = inside_samples_coords[:, 1]
    axes.scatter(x, y, s=marker_size, c='r', alpha=marker_alpha, label='inside')


plt.show()
