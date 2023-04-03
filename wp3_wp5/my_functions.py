# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:13:42 2022

@author: Monica
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import gopython
import os
from io import StringIO


def HPLC_simulation(theta, U):
    """
    gPROMS simulation of an HPLC system.
    Model: Equilibrium Dispersive Model (EDM) with Linear Adsorption Isotherm
    Assumption: ideal rectangular injection profile
    
    Inputs: theta: array of parameters
            U: array of control variables
        
    Output: time: array of times
            Cout: concentration profile at the column outlet
    """

    a0, Ct, Ss = theta
    Ct = theta[1] * 100
    temperature, init_SS = U
    
    # Define other variables required by gPROMS
    NETP = 10000
    flowrate = 0.5
    Sample_conc_analyte = 1.55
    Sample_vol = 0.001
    final_SS = init_SS
    gradient_time = 0
    gradient_initial_hold_time = 100000
    volume_sampleinj_column = 0
    volume_gradientgen_column = 0
    porosity_external = 0.383
    porosity_total = 0.645
    particle_diam = 0.0005
    fname = "EDM_process"

    # Move to the directory containing the gPROMS file
    os.chdir("gPROMS_file")

    # Start gOPython
    status = gopython.start(fname, fname, fname)

    # Simulate
    status, result = gopython.evaluate([a0, Ss, Ct, NETP, Sample_conc_analyte, Sample_vol,
                                        flowrate, init_SS, final_SS, gradient_initial_hold_time, gradient_time,
                                        temperature, volume_sampleinj_column,
                                        volume_gradientgen_column, porosity_external, porosity_total, particle_diam])

    # Close gOPython
    gopython.stop()

    # Convert the gPLOT to array variable
    var_name, outputdata = gp2py(fname)

    # Convert into the desired format
    time = outputdata[:, 0]
    Cout = outputdata[:, 1]
    
    os.chdir("General/")

    return  time, Cout 

def gp2py(fname):
    """
    This function convert the gPLOT file to a numpy array.

    :param:
    fname:      name of the process

    :return:
    var_name:   the name of the monitored variables
    outputdata: a numpy array containing, by column, the time and the monitored variables
    """
    
    filepath = "output/" + fname + ".gPLOT"

    with open(filepath, 'r') as f:
        txt = f.read()

    data = pd.read_csv(StringIO(txt), sep='\r\n', header=None, engine='python')
    nvars = int(data[0][0])
    var_name = data[0][1:1 + nvars]
    outputdata = data[0][1 + nvars:-1].values
    outputdata = np.reshape(outputdata, (-1, nvars + 1)).astype(float)
    return var_name, outputdata
