#####################################################################
# IMPORTS
#####################################################################

import json, sys, inspect
import numpy as np
from csv import writer
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

from datahandle import data_read
from mathfunk import singlefit, doublefit
from mathfunk import set_resolution, get_chisq


#####################################################################
# CONSTANTS
#####################################################################




#####################################################################
# PRINTOUT STRINGS
#####################################################################
MISSING_KEYWORD = 'ERROR: keyword not present in the configuration file'


#####################################################################
# GET THE INPUT
#####################################################################


# Input config file as CLI argument
print("\n---> Opening configuration file...\n")
try:
    configuration_file_path = sys.argv[1]
    with open(configuration_file_path, 'r') as f:
        cfg = json.load(f)
except Exception as e:
    print(e)

# Read the data with the method from datahandle.py
data = data_read(configuration_file_path)



#####################################################################
# PRELIMINARY FITTING AND CORRELATIONS
#####################################################################


#----------- Check for booked presets:
try:
    levo_booked = True if 'levoglucosan' in cfg['presets'] else False
except NameError as e:
    print(MISSING_KEYWORD, e)
    # Fill in with other presets
##### TODO implement user-defined fits and correlations (for the GUI)

alpha_BC, alpha_FF, alpha_WB = cfg['alpha BC'], cfg['alpha FF'], cfg['alpha WB']
best_alpha_BC, best_alpha_FF, best_alpha_WB = alpha_BC, alpha_FF, alpha_WB

#----------- Carry out the LEVOGLUCOSAN CORRELATION ADJUSTMENT
if levo_booked:
    print(f'---> Performing correlation maximisation with'
            ' "levoglucosan" preset...\n')
    # Set two parameters
    for iteration_number in range(cfg['iterations']):
        print(f"\t*Iteration {iteration_number + 1}")
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

    #--- From now on, use alpha_XX for the best values
    #--- or the default values if no optimizatios was done
    alpha_BC, alpha_FF, alpha_WB = best_alpha_BC, best_alpha_FF, best_alpha_WB
    print(f"\nThe best parameters for the the correlation with levoglucosan are "
            f"(alpha_BC = {round(alpha_BC, 2)}, alpha_FF = {round(alpha_FF, 2)}, alpha_WB = {round(alpha_WB, 2)}).\n")

    # TODO produce nice plots for this procedure




##################################################################
# FITTING PROCEDURE WITH OPTIMIZED PARAMETERS
##################################################################

print(f'---> Fitting the experimental data...\n')

for sample in data:

    #--------- AAE fit
    prp = sample.properties
    aae_fitres = curve_fit(singlefit, prp.wavelength,
            prp.abs, p0=(1e5, 1), bounds=([1,0.5], [1e15, 3]),
            sigma=prp.u_abs)
    # Save the exponent
    prp.aae = aae_fitres[0][1]
    prp.u_aae = np.sqrt(aae_fitres[1][1][1])
    #--------- Second AAE fit fixing the exp for lower uncertainty
    def singlefit_fix(x, scale):
        return singlefit(x, scale, prp.aae)
    second_aae_fitres = curve_fit(singlefit_fix, prp.wavelength,
            prp.abs, p0=(1e5), bounds=(1, 1e15),
            sigma=prp.u_abs)
    # Save the scale
    prp.scale = second_aae_fitres[0][0]
    prp.u_scale = np.sqrt(second_aae_fitres[1][0][0])
    # Calculates the chisquared for this fit
    expected = [singlefit_fix(x, prp.scale) for x in prp.wavelength]
    ndf = len(prp.wavelength) - 2
    tmp, prp.red_chisq_aae_fit = get_chisq(prp.abs, expected,
            prp.u_abs, ndf)

    #--------- Components fit
    def typefit(x, A, B, alpha_BrC):
        """Fix alpha BC to the best value"""
        return doublefit(x, A, alpha_BC, B, alpha_BrC)
    type_fitres = curve_fit(typefit, prp.wavelength,
            prp.abs, p0=(1e3, 1e10, 3),
            bounds=([1, 1, 1], [1e15, 1e15, 10]),
            sigma=prp.u_abs)
    # Save alpha_BrC
    prp.alpha_brc = type_fitres[0][2]
    prp.u_alpha_brc = np.sqrt(type_fitres[1][2][2])
    #------ Improved components fit
    def typefit_fix(x, A, B):
        """Fix alpha BrC for uncertainty improvement"""
        return typefit(x, A, B, prp.alpha_brc)
    second_type_fitres = curve_fit(typefit_fix, prp.wavelength,
            prp.abs, p0=(1e3, 1e10),
            bounds=([1, 1], [1e15, 1e15]),
            sigma=prp.u_abs)
    prp.A = type_fitres[0][0]
    prp.u_A = np.sqrt(type_fitres[1][0][0])
    prp.B = type_fitres[0][1]
    prp.u_B = np.sqrt(type_fitres[1][1][1])
    # Calculates the chisquared for this fit
    expected = [typefit_fix(x, prp.A, prp.B) for x in prp.wavelength]
    ndf = len(prp.wavelength) - 3
    tmp, prp.red_chisq_type_fit = get_chisq(prp.abs, expected,
            prp.u_abs, ndf)

    #---- Source fit
    def sourcefit(x, A_p, B_p):
        """Fix alpha_FF and alpha_WB to the best values"""
        return doublefit(x, A_p, alpha_FF, B_p, alpha_WB)
    source_fitres = curve_fit(sourcefit, prp.wavelength,
            prp.abs, p0=(1e3, 1e10),
            bounds=([1, 1], [1e15, 1e15]),
            sigma=prp.u_abs)
    prp.A_p = source_fitres[0][0]
    prp.u_A_p = np.sqrt(source_fitres[1][0][0])
    prp.B_p = source_fitres[0][1]
    prp.u_B_p = np.sqrt(source_fitres[1][1][1])
    # Calculates the chisquared for this fit
    expected = [sourcefit(x, prp.A_p, prp.B_p) for x in prp.wavelength]
    ndf = len(prp.wavelength) - 2
    tmp, prp.red_chisq_source_fit = get_chisq(prp.abs, expected,
            prp.u_abs, ndf)

             

#####################################################################
# FIT PARAMETERS WRITEOUT
#####################################################################

try:
    print(f"---> Writing fit results to {cfg['fit output']}\n")
    out_path = cfg['fit output']
    with open(out_path, 'w') as f:
        writa = writer(f)
        header = ['Name', 'Scale', 'eScale', 'AAE', 
                'eAAE', 'Red_chisq', 'A', 'eA', 'B', 
                'eB', 'alpha_BrC', 'ealpha_BrC', 'Red_chisq', 
                'A\'', 'eA\'', 'B\'', 'eB\'', 'Red_chisq']
        writa.writerow(header)
        for sample in data:
            prp = sample.properties
            linetowrite = [sample.name, prp.scale, prp.u_scale, 
                    prp.aae, prp.u_aae, prp.red_chisq_aae_fit, 
                    prp.A, prp.u_A, prp.B, prp.u_B, prp.alpha_brc,
                    prp.u_alpha_brc, prp.red_chisq_type_fit, 
                    prp.A_p, prp.u_A_p, prp.B_p, prp.u_B_p, 
                    prp.red_chisq_source_fit]
            writa.writerow(linetowrite) 
except KeyError as ke:
    print(MISSING_KEYWORD, ke)


# TODO at the end of everything write a .log file with
# all the analysis parameter




#####################################################################
# SAVE PLOTS
#####################################################################

#---- Save fit plots plots for the two fits if booked
if cfg['fit plots']:
    print(f'---> Saving fit plots in {cfg["working directory"]}' + f'plots/fitplots/') 
    for sample in data:
        prp = sample.properties
        # Data points
        plt.errorbar(prp.wavelength, prp.abs, 
              xerr=prp.u_wavelength, yerr=prp.u_abs,
              fmt='.k', elinewidth=0.8, markersize=1.2,
              label=f'data {sample.name}')
        x = np.linspace(prp.wavelength[0], prp.wavelength[-1], 500)
        # Type fit
        ytype = typefit_fix(x, prp.A, prp.B)
        plt.plot(x, ytype, 'r', 
              label='Component fit (BC + BrC)')
        ybc = singlefit(x, prp.A, alpha_BC)
        plt.plot(x, ybc, 'k', linewidth=0.5,
              label='BC contribution')
        ybrc = singlefit(x, prp.B, prp.alpha_brc)
        plt.plot(x, ybrc, 'y', linewidth=0.5,
                label='BrC contribution')
        # Source fit
        ysource = sourcefit(x, prp.A_p, prp.B_p)
        plt.plot(x, ysource, 'b', linestyle='dashed', 
              label='Source fit (FF + WB)')
        yff = singlefit(x, prp.A_p, alpha_FF)
        plt.plot(x, yff, 'm', linewidth=0.5, linestyle='dashed',
              label='FF contribution')
        ywb = singlefit(x, prp.B_p, alpha_WB)
        plt.plot(x, ywb, 'g', linewidth=0.5, linestyle='dashed',
                label='WB contribution')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel(r'Absorption coefficient, $b_{abs}$  ' + '[Mm'+ r'$^{-1}$' + ']') if prp.data_type == 'Babs' else plt.ylabel('100 ABS')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg['working directory'] + f'plots/fitplots/{sample.name}.png', dpi = 300)
        plt.close()
