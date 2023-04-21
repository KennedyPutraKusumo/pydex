# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:26:06 2022

@author: Monica
"""

# Import my function
#

from matplotlib import pyplot as plt
from my_functions import *
from pydex.core.designer import Designer

def pydex_sim(ti_controls, sampling_times, model_parameters):
    inner_designer = Designer()
    return_sensitivities = inner_designer.detect_sensitivity_analysis_function()

    print("TI CONTROLS:")
    print(ti_controls)
    print("MODEL PARAMETERS:")
    print(model_parameters)
    if not return_sensitivities:
        try:
            y = np.zeros((1001, 1))
            t, y_sim = HPLC_simulation(model_parameters, ti_controls)
            y[:y_sim.shape[0], 0] = y_sim
        except ValueError:
            print("TI CONTROLS:")
            print(ti_controls)
            print("MODEL PARAMETERS:")
            print(model_parameters)
        return y[:, None]
    else:
        try:
            y = np.full((1001, 1), np.nan)
            sens = np.full((1001, 1, 3), np.nan)
            t, y_sim, sens_sim = HPLC_simulation(model_parameters, ti_controls)
            y[:y_sim.shape[0], 0] = y_sim
            sens[:sens_sim.shape[0], 0, :] = sens_sim
        except ValueError:
            print("TI CONTROLS:")
            print(ti_controls)
            print("MODEL PARAMETERS:")
            print(model_parameters)
        return y, sens

if __name__ == '__main__':

    # Initial conditions
    flowrate = 1  # mobile phase flowrate in mL/min
    temperature = 25  # degree Celsius
    amount_ACN = 0.14  # amount of organic modifier at the beginning of the gradient (from 0 to 1)
    U = np.array([temperature, amount_ACN]).T

    # Adsorption isotherm parameters
    a0 = 40
    ct = 12
    Ss = 8

    theta = np.array([a0, ct, Ss]).T
    y, sens = pydex_sim(U, np.array([]), theta)

    # Perform the gPROMS simulation
    time, C_out, sens = HPLC_simulation(theta, U)

    # visualize outputs
    plt.plot(time, C_out, 'r')
    fig = plt.figure(figsize=(15, 8))
    axes1 = fig.add_subplot(131)
    axes2 = fig.add_subplot(132)
    axes3 = fig.add_subplot(133)
    axes1.plot(time, sens[:, 0])
    axes2.plot(time, sens[:, 1])
    axes3.plot(time, sens[:, 2])
    fig.tight_layout()

    reso = 3j
    a0_grid = np.mgrid[35:45:reso]
    ct_grid = np.mgrid[11:13:reso]
    Ss_grid = np.mgrid[7:9:reso]

    fig = plt.figure()
    fig2 = plt.figure()
    axes1 = fig.add_subplot(131)
    axes4 = fig2.add_subplot(131)
    axes5 = fig2.add_subplot(132)
    axes6 = fig2.add_subplot(133)
    for a0 in a0_grid:
        theta = np.array([a0, ct, Ss]).T
        time, C_out, sens = HPLC_simulation(theta, U)
        axes1.plot(
            time,
            C_out,
            label=f"a0: {a0:.2f}",
        )
        axes4.plot(
            time,
            sens[:, 0],
            label=f"a0: {a0:.2f}",
        )
        axes4.set_ylabel("Sens of a0")
        axes4.set_xlabel("Time (mins)")
        axes5.plot(
            time,
            sens[:, 1],
            label=f"a0: {a0:.2f}",
        )
        axes5.set_ylabel("Sens of CT")
        axes5.set_xlabel("Time (mins)")
        axes6.plot(
            time,
            sens[:, 2],
            label=f"a0: {a0:.2f}",
        )
        axes6.set_ylabel("Sens of Ss")
        axes6.set_xlabel("Time (mins)")
    a0 = 40
    axes2 = fig.add_subplot(132)
    for ct in ct_grid:
        theta = np.array([a0, ct, Ss]).T
        time, C_out, sens = HPLC_simulation(theta, U)
        axes2.plot(
            time,
            C_out,
            label=f"ct: {ct:.2f}",
        )
        axes4.plot(
            time,
            sens[:, 0],
            label=f"ct: {ct:.2f}",
        )
        axes4.set_ylabel("Sens of a0")
        axes4.set_xlabel("Time (mins)")
        axes5.plot(
            time,
            sens[:, 1],
            label=f"ct: {ct:.2f}",
        )
        axes5.set_ylabel("Sens of CT")
        axes5.set_xlabel("Time (mins)")
        axes6.plot(
            time,
            sens[:, 2],
            label=f"ct: {ct:.2f}",
        )
        axes6.set_ylabel("Sens of Ss")
        axes6.set_xlabel("Time (mins)")
    ct = 12
    axes3 = fig.add_subplot(133)
    for Ss in Ss_grid:
        theta = np.array([a0, ct, Ss]).T
        time, C_out, sens = HPLC_simulation(theta, U)
        axes3.plot(
            time,
            C_out,
            label=f"Ss: {Ss:.2f}",
        )
        axes4.plot(
            time,
            sens[:, 0],
            label=f"Ss: {Ss:.2f}",
        )
        axes4.set_ylabel("Sens of a0")
        axes4.set_xlabel("Time (mins)")
        axes5.plot(
            time,
            sens[:, 1],
            label=f"Ss: {Ss:.2f}",
        )
        axes5.set_ylabel("Sens of CT")
        axes5.set_xlabel("Time (mins)")
        axes6.plot(
            time,
            sens[:, 2],
            label=f"Ss: {Ss:.2f}",
        )
        axes6.set_ylabel("Sens of Ss")
        axes6.set_xlabel("Time (mins)")
    Ss = 8
    axes1.legend()
    axes2.legend()
    axes3.legend()
    axes4.legend()
    axes5.legend()
    axes6.legend()
    fig.tight_layout()
    plt.show()
    print('hi')
