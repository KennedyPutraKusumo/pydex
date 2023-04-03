# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:26:06 2022

@author: Monica
"""

# Import my function
#

import scipy.optimize
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from my_functions import *

def pydex_sim(ti_controls, sampling_times, model_parameters):
    print("TI CONTROLS:")
    print(ti_controls)
    print("MODEL PARAMETERS:")
    print(model_parameters)
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

    # Perform the gPROMS simulation
    time, C_out = HPLC_simulation(theta, U)
    plt.plot(time, C_out, 'r')
    plt.show()

    reso = 3j
    a0_grid = np.mgrid[35:45:reso]
    ct_grid = np.mgrid[11:13:reso]
    Ss_grid = np.mgrid[7:9:reso]

    fig = plt.figure()
    axes1 = fig.add_subplot(131)
    for a0 in a0_grid:
        theta = np.array([a0, ct, Ss]).T
        time, C_out = HPLC_simulation(theta, U)
        axes1.plot(
            time,
            C_out,
            label=f"a0: {a0:.2f}",
        )
    a0 = 40
    axes2 = fig.add_subplot(132)
    for ct in ct_grid:
        theta = np.array([a0, ct, Ss]).T
        time, C_out = HPLC_simulation(theta, U)
        axes2.plot(
            time,
            C_out,
            label=f"ct: {ct:.2f}",
        )
    ct = 12
    axes3 = fig.add_subplot(133)
    for Ss in Ss_grid:
        theta = np.array([a0, ct, Ss]).T
        time, C_out = HPLC_simulation(theta, U)
        axes3.plot(
            time,
            C_out,
            label=f"Ss: {Ss:.2f}",
        )
    Ss = 8
    axes1.legend()
    axes2.legend()
    axes3.legend()
    plt.show()
    print('hi')
