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

def pydex_sim_dummy(ti_controls, sampling_times, model_parameters):
    y = np.zeros((1001, 1))
    return y[:, None]

if __name__ == '__main__':

    print('hi')
