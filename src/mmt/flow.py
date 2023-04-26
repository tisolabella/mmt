#####################################################################
# IMPORTS
#####################################################################

import json, sys, inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

from datahandle import data_read
from mathfunk import singlefit, doublefit, set_resolution


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
print("\n---> Opening configuration file...\n")
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
    print("\n---> Preliminary fitting: searching the parameter space...")
    levo_booked = True if 'levoglucosan' in config['presets'] else False
    # Fill in with other presets
##### TODO implement user-defined fits and correlations (for the GUI)

# Carry out the LEVOGLUCOSAN CORRELATION ADJUSTMENT
if levo_booked:
    # Set two parameters
    alpha_FF, alpha_WB = 1.0, 2.0
    best_alpha_BC, best_alpha_FF, best_alpha_WB = 1.0, 1.0, 2.0
    # List for saving correlations TODO remove - unnecessary since the lists
    # are created with mathfunk.set_resolution()
    alpha_BC_set = config['alpha BC values']
    alpha_FF_set = config['alpha FF values']
    alpha_WB_set = config['alpha WB values']
    for iteration_number in range(config['iterations']):
        print(f"Iteration number {iteration_number + 1}")
        # Set the search resolution
        alpha_BC_set, alpha_FF_set, alpha_WB_set = set_resolution(best_alpha_BC, 
                best_alpha_FF, best_alpha_WB, iteration=iteration_number)
        #------ Do the alpha_BC iteration -------------
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
        max_R2_index = R_2_alpha_BC.index(max(R_2_alpha_BC))
        best_alpha_BC = alpha_BC_set[max_R2_index]
        # For checking
        #print(alpha_BC_set, R_2_alpha_BC)
        #print("best alpha BC", best_alpha_BC, " for R2 ", R_2_alpha_BC[max_R2_index])
        #plt.plot(alpha_BC_set, R_2_alpha_BC)
        #plt.show()

        #------ Do the alpha_FF iteration -------------
        # List for the correlations
        R_2_alpha_FF = []
        for alpha_FF in alpha_FF_set:
            # List to save the BC_WB(lambda_long) and levo
            BC_WB_set, levo_set = [], []
            # Fix the alpha BC value in a new function 
            def typefit(x, A, B, alpha_BrC):
                return doublefit(x, A, best_alpha_BC, B, alpha_BrC)
            # Fix the alpha FF value in a new function 
            def sourcefit(x, A_p, B_p):
                return doublefit(x, A_p, alpha_FF, B_p, alpha_WB)
            # Iterate over all the samples
            for sample in data:
                prop = sample.properties
                # Do fit
                try:
                    typefitres = curve_fit(typefit, prop.wavelength,
                            prop.abs, p0=(1e3, 1e10, 3),
                            bounds=([1, 1, 1], [1e15, 1e15, 10]),
                            sigma=prop.u_abs)
                    sourcefitres = curve_fit(sourcefit, prop.wavelength,
                            prop.abs, p0=(1e3, 1e10),
                            bounds=([1, 1], [1e15, 1e15]),
                            sigma=prop.u_abs)
                    A = typefitres[0][0]
                    A_p = sourcefitres[0][0]
                    # Apportion BC WB at the longest wavelength
                    lambda_long =  max(prop.wavelength) 
                    BC_WB = (A - A_p) * lambda_long ** ( - best_alpha_BC) 
                    BC_WB_set.append(BC_WB)
                except Exception as e:
                    print(f'FIT ERROR for ALPHA_FF: {e}')
                    BC_WB_set.append(0)
                levo_set.append(prop.Levoglucosan)
            # Calculate regression and append R^2
            try:
                regression_res = linregress(levo_set, y=BC_WB_set)
                R_2_alpha_FF.append(regression_res.rvalue ** 2)
            except Exception as e:
                print(f'REGRESSION ERROR for ALPHA FF: {e}')
                R_2_alpha_FF.append(0)
        max_R2_index = R_2_alpha_FF.index(max(R_2_alpha_FF))
        best_alpha_FF = alpha_FF_set[max_R2_index]
        # For checking
        #print(alpha_FF_set, R_2_alpha_FF)
        #print("best alpha FF", best_alpha_FF, " for R2 ", R_2_alpha_FF[max_R2_index])
        #plt.plot(alpha_FF_set, R_2_alpha_FF)
        #plt.show()

        #------ Do the alpha_WB iteration -------------
        # List for the correlations
        R_2_alpha_WB = []
        for alpha_WB in alpha_WB_set:
            # List to save the BC_WB(lambda_long) and levo
            BC_WB_set, levo_set = [], []
            # Fix the alpha BC value in a new function 
            def typefit(x, A, B, alpha_BrC):
                return doublefit(x, A, best_alpha_BC, B, alpha_BrC)
            # Fix the alpha FF value in a new function 
            def sourcefit(x, A_p, B_p):
                return doublefit(x, A_p, best_alpha_FF, B_p, alpha_WB)
            # Iterate over all the samples
            for sample in data:
                prop = sample.properties
                # Do fit
                try:
                    typefitres = curve_fit(typefit, prop.wavelength,
                            prop.abs, p0=(1e3, 1e10, 3),
                            bounds=([1, 1, 1], [1e15, 1e15, 10]),
                            sigma=prop.u_abs)
                    sourcefitres = curve_fit(sourcefit, prop.wavelength,
                            prop.abs, p0=(1e3, 1e10),
                            bounds=([1, 1], [1e15, 1e15]),
                            sigma=prop.u_abs)
                    A = typefitres[0][0]
                    A_p = sourcefitres[0][0]
                    # Apportion BC WB at the longest wavelength
                    lambda_long =  max(prop.wavelength) 
                    BC_WB = (A - A_p) * lambda_long ** ( - best_alpha_BC) 
                    BC_WB_set.append(BC_WB)
                except Exception as e:
                    print(f'FIT ERROR for ALPHA_FF: {e}')
                    BC_WB_set.append(0)
                levo_set.append(prop.Levoglucosan)
            # Calculate regression and append R^2
            try:
                regression_res = linregress(levo_set, y=BC_WB_set)
                R_2_alpha_WB.append(regression_res.rvalue ** 2)
            except Exception as e:
                print(f'REGRESSION ERROR for ALPHA WB: {e}')
                R_2_alpha_WB.append(0)
        max_R2_index = R_2_alpha_WB.index(max(R_2_alpha_WB))
        best_alpha_WB = alpha_WB_set[max_R2_index]
        # For checking
        #print(alpha_WB_set, R_2_alpha_WB)
        #print("best alpha WB", best_alpha_WB, " for R2 ", R_2_alpha_WB[max_R2_index])
        #plt.plot(alpha_WB_set, R_2_alpha_WB)
        #plt.show()
    alpha_BC, alpha_FF, alpha_WB = best_alpha_BC, best_alpha_FF, best_alpha_WB
    print(f"\nThe best parameters for the the correlation with levoglucosan are "
            f"(alpha_BC = {round(alpha_BC, 2)}, alpha_FF = {round(alpha_FF, 2)}, alpha_WB = {round(alpha_WB, 2)}).")

