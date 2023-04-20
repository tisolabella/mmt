#####################################################################
# IMPORTS
#####################################################################

import json, sys, inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

from datahandle import data_read
from mathfunk import singlefit, doublefit


#####################################################################
# CONSTANTS
#####################################################################




#####################################################################
# PRINTOUT STRINGS
#####################################################################



#####################################################################
# GET THE INPUT
#####################################################################


# Input config file as CLI argument
try:
    configuration_file_path = sys.argv[1]
    with open(configuration_file_path, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(e)

# Read the data with the method from datahandle.py
data = data_read(configuration_file_path)



#####################################################################
# PRELIMINARY FITTING AND CORRELATIONS
#####################################################################


# Check for booked presets:
if len(config['presets']) > 0:
    levo_booked = True if 'levoglucosan' in config['presets'] else False
    # Fill in with other presets
##### TODO implement user-defined fits and correlations (for the GUI)

# Carry out the LEVOGLUCOSAN CORRELATION ADJUSTMENT
if levo_booked:
    # Set two parameters
    alpha_FF, alpha_WB = 1.0, 2.0
    # List for saving correlations
    alpha_BC_set = config['alpha BC values']
    for iteration_number in range(config['iterations']):
        # List for the correlations
        R_2_alpha_BC = []
        for alpha_BC in alpha_BC_set:
            # List to save the BrC(lambda_short) and levo
            BrC_set, levo_set = [], []
            # Fix the alpha BC value in a new function
            def typefit(x, A, B, alpha_BrC):
                return doublefit(x, A, alpha_BC, B, alpha_BrC)
            # Iterate over all the samples
            for sample in data:
                prop = sample.properties
                # Do fit
                try:
                    fitres = curve_fit(typefit, prop.wavelength,
                            prop.abs, p0=(1e3, 1e10, 3),
                            bounds=([1, 1, 1], [1e15, 1e15, 10]),
                            sigma=prop.u_abs)
                    A = fitres[0][0]
                    B = fitres[0][1]
                    alpha_BrC = fitres[0][2]
                    # Apportion BrC at the shortest wavelength
                    lambda_short =  min(prop.wavelength) 
                    BrC = B * lambda_short ** ( - alpha_BrC) 
                    BrC_set.append(BrC)
                except Exception as e:
                    print(f'FIT ERROR for ALPHA_BC: {e}')
                    BrC_set.append(0)
                levo_set.append(prop.Levoglucosan)
            # Calculate regression and append R^2
            try:
                regression_res = linregress(levo_set, y=BrC_set)
                R_2_alpha_BC.append(regression_res.rvalue ** 2)
            except Exception as e:
                print(f'REGRESSION ERROR for ALPHA BC: {e}')
                R_2_alpha_BC.append(0)
        # For checking
        #print(alpha_BC_set, R_2_alpha_BC)
        #plt.plot(alpha_BC_set, R_2_alpha_BC)
        #plt.show()
        # It works so far!
    







