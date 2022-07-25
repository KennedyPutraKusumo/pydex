from matplotlib import pyplot as plt
import numpy as np


""" RESPONSE """
def y_response(x, theta1):
    theta_0 = 5.5
    return theta_0 * np.exp(theta1 * x)

x_res = 51
theta1_res = 600

x_list = np.linspace(0, 0.5, x_res)
theta1_list = np.linspace(-10.0, -7.5, np.round(theta1_res/4).astype(int))
outer_y = []
for theta1 in theta1_list:
    inner_y = []
    for x in x_list:
        inner_y.append(y_response(x, theta1))
    outer_y.append(inner_y)
outer_y = np.asarray(outer_y).T

alpha = 0.05

figsize = (4.5, 3.5)
fig = plt.figure(figsize=figsize)
axes = fig.add_subplot(111)
# axes.fill_between(
#     np.asarray(x_list),
#     np.min(outer_y, axis=1),
#     np.max(outer_y, axis=1),
#     facecolor="tab:red",
#     alpha=0.5,
# )
axes.plot(
    np.asarray(x_list),
    outer_y,
    c="tab:red",
    alpha=alpha,
)
axes.plot(
    np.asarray(x_list),
    np.mean(outer_y, axis=1),
    c="tab:red",
    ls="--",
)
theta1_list = np.linspace(-7.5, 0.0, int(theta1_res*3/4))
outer_y_group2 = []
for theta1 in theta1_list:
    inner_y = []
    for x in x_list:
        inner_y.append(y_response(x, theta1))
    outer_y_group2.append(inner_y)
outer_y_group2 = np.asarray(outer_y_group2).T

# axes.fill_between(
#     np.asarray(x_list),
#     np.min(outer_y_group2, axis=1),
#     np.max(outer_y_group2, axis=1),
#     facecolor="tab:green",
#     alpha=0.5,
# )
axes.plot(
    np.asarray(x_list),
    outer_y_group2,
    c="tab:green",
    alpha=alpha,
)
axes.plot(
    np.asarray(x_list),
    np.mean(outer_y_group2, axis=1),
    c="tab:green",
    ls="--",
)
axes.set_xlabel("Experimental Variable $x \ (\cdot)$ ")
axes.set_ylabel(r"Response $y \ (\cdot)$")

fig.tight_layout()
fig.savefig("exp_model_response_shade.png", dpi=360)

""" SENSITIVITIES """
def dydtheta1(x, theta1):
    theta_0 = 5.5
    return theta_0 * x * np.exp(theta1 * x)

x_res = 51
theta1_res = 500

x_list = np.linspace(0, 0.5, x_res)
theta1_list = np.linspace(-10.0, -7.5, np.round(theta1_res/4).astype(int))
outer_y = []
for theta1 in theta1_list:
    inner_y = []
    for x in x_list:
        inner_y.append(dydtheta1(x, theta1))
    outer_y.append(inner_y)
outer_y = np.asarray(outer_y).T

alpha = 0.05

fig = plt.figure(figsize=figsize)
axes = fig.add_subplot(111)
# axes.fill_between(
#     np.asarray(x_list),
#     np.min(outer_y, axis=1),
#     np.max(outer_y, axis=1),
#     facecolor="tab:red",
#     alpha=0.5,
# )
axes.plot(
    np.asarray(x_list),
    outer_y,
    c="tab:red",
    alpha=alpha,
)
axes.plot(
    np.asarray(x_list),
    np.mean(outer_y, axis=1),
    c="tab:red",
    ls="--",
)
theta1_list = np.linspace(-7.5, 0.0, int(theta1_res*3/4))
outer_y_group2 = []
for theta1 in theta1_list:
    inner_y = []
    for x in x_list:
        inner_y.append(dydtheta1(x, theta1))
    outer_y_group2.append(inner_y)
outer_y_group2 = np.asarray(outer_y_group2).T

# axes.fill_between(
#     np.asarray(x_list),
#     np.min(outer_y_group2, axis=1),
#     np.max(outer_y_group2, axis=1),
#     facecolor="tab:green",
#     alpha=0.5,
# )
axes.plot(
    np.asarray(x_list),
    outer_y_group2,
    c="tab:green",
    alpha=alpha,
)
axes.plot(
    np.asarray(x_list),
    np.mean(outer_y_group2, axis=1),
    c="tab:green",
    ls="--",
)
axes.set_xlabel("Experimental Variable $x \ (\cdot)$ ")
axes.set_ylabel(r"Sensitivity $\frac{\partial y}{\partial \theta_1} \ (\cdot)$")

fig.tight_layout()
fig.savefig("exp_model_sensitivity_shade.png", dpi=360)

plt.show()
