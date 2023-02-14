# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:26:06 2022

This script allows to:
 1) map the information provided by each experiment.
 2) estimate the optimal parameter considering the most informative experiments.
 3) validate the model against the less informative experiments.

@author: Monica
"""

# Import my function
#

import scipy.optimize
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from my_functions import *

# Initial conditions
flowrate = 1  # mobile phase flowrate in mL/min
temperature = 25  # degree Celsius
amount_ACN = 0.14  # amount of organic modifier at the beginning of the gradient (from 0 to 1)
U = np.array([flowrate, temperature, amount_ACN]).T

# Adsorption isotherm parameters
a0 = 20
ct = 1200
Ss = 8

theta = np.array([a0, ct, Ss]).T

# Perform the gPROMS simulation
time, C_out = HPLC_simulation(theta, U)
plt.plot(time, C_out, 'r')
plt.show()
print('hi')
