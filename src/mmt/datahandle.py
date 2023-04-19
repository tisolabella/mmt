###########################################################
# IMPORTS
###########################################################

import sys
import json
import pandas as pd
import numpy as np
from pprint import pprint


###########################################################
# NUMEERICAL CONSTANTS
###########################################################
DEFAULT_ABS_UNCERTAINTY = .5



###########################################################
# PRINTOUT STRINGS
###########################################################
INPUT_FILE_KEY_NOT_PRESENT = "CONFIGURATION FILE ERROR: the key 'input file' was not found in the configuration file"
INPUT_FILE_NO_ABS = "INPUT FILE ERROR: the input file does not contain either a column named ABS nor a column named Babs"
MASS_APPO_NO_EC_OC = "WARNING: the mass_appo flag has been set to True but the EC and OC values were not provided in the input  file. The mass apportionment will not be performed."
MASS_APPO_NO_FLAG = "WARNING: EC and OC were provided but the mass_appo flag has not been set to True. Mass apportionment will not be performed."



###########################################################
# CLASS DEFINITIONS
###########################################################

class Sample():
    """Represents an individual sample under analysis"""
    def __init__(self, name):
        self.name = name
        self.properties = Property()
    def __str__(self):
        # Name string
        ns = f'Sample object - Name {self.name}'
        # Properties string
        ps =str(list(self.properties.__dict__.keys()))
        string_to_print = ns + '\nProperties - ' + ps +'\n'
        return string_to_print

class Property():
    """Container object for all the numerical properties of the sample."""
    def __init__(self):
        """The properties are:

        # Input data
        type_of_data (string, either "ABS" or "Babs")
        wavelength (list, nm)
        u_wavelength (list, nm, corresponding to wavelength)
        abs (list, corresponding to wavelength)
        u_abs (list, corresponding to wavelength)
        ec (float, ug/cm3)
        oc (float, ug/cm3)

        # From the fits
        aae (float)
        u_aae (float)
        scale (float)
        u_scale (float)
        chisq_aae_fit (float)
        red_chisq_aae_fit (float)
        a (float)
        u_a (float)
        b (float)
        u_b (float)
        alpha_brc (float)
        u_alpha_brc (float)
        chisq_type_fit (float)
        red_chisq_type_fit (float)
        a_p (float)
        u_a_p (float)
        b_p (float)
        u_b_p (float)
        chisq_source_fit (float)
        red_chisq_source_fit (float)

        # From the optical apportionment
        brc (list, corresponding to wavelength)
        brc_fraction (list, corresponding to wavelength)
        bc_ff (list, corresponding to wavelength)
        bc_ff_fraction (list, corresponding to wavelength)
        bc_wb (list, corresponding to wavelength)
        bc_wb_fraction (list, corresponding to wavelength)

        # From the mass apportionment
        ec_ff (float)
        ec_ff_fraction (float)
        ec_wb (float)
        ec_wb_fraction (float)
        oc_ff (float)
        oc_ff_fraction (float)
        oc_wb (float)
        oc_wb_fraction (float)
        oc_nc (float)
        oc_nc_fraction (float)

        """
        pass





###########################################################
# DATA READING SECTION
###########################################################

# Function to call for reading the data
def data_read(configuration_file):
    """Read the data from an input file.

    Parameters:
        configuration_file : str,
            the path to the JSON configuration file which contains, 
            among other information, the input data file path under the 
            key 'input file'.

    Returns:
        data : sequence,
            a list of Sample objects, each one corresponding to a physical
            sample for analysis.
    """
    # Open the input data file
    try:
        input_file = configuration_file['input file']
        # For additional measurements on data, e.g. levoglucosan
        additional_measurements = split(configuration_file['additional'], ',')
        rawdata = pd.read_csv(input_file)
    except KeyError as e:
        print(e)
        print(INPUT_FILE_KEY_NOT_PRESENT)
    # Read the names, the wavelengths and the absorbance (or babs) values
    try:
        names = [x for x in rawdata['Name']]
        wavelengths = [float(x) for x in rawdata['Wavelength']]
        if 'Babs' in rawdata.keys():
            b_abs = [float(x) for x in rawdata['Babs']]
            type_of_data = 'Babs'
        elif 'ABS' in rawdata.keys():
            ABS = [float(x) for x in rawdata['ABS']]
            type_of_data = 'ABS'
        else:
            print(INPUT_FILE_NO_ABS)
    except NameError as e:
        print(e)
    # Check if babs uncertainty is provided and read it
    uncertainty_provided = True if 'Uncertainty' in rawdata.keys() else False
    # Check if a mass apportionment is required
    mass_appo_requested = True if 'EC' in rawdata.keys() and 'OC' in rawdata.keys() else False
    try:
        if configuration_file['mass_appo'] and not mass_appo_requested:
            print(MASS_APPO_NO_EC_OC)
        if not configuration_file['mass_appo'] and mass_appo_requested:
            print(MASS_APPO_NO_FLAG)
            mass_appo = False
    except KeyError as ke:
        print(ke)
    # Make the names unique instead of repeated
    names = np.unique(np.array(names)).tolist()
    # Create the list of sample objects. This will be the output
    data = [Sample(name) for name in names]
    # Fill the Sample.property 
    for name in names:
        # Scan for wavelengths and ABS or Babs
        # Temporary lists to populate from file
        tmp_wlength, tmp_u_wlength, tmp_abs, tmp_u_abs, tmp_ec, tmp_oc = [], [], [], [], [], [] 
        for n, w, a in zip(rawdata['Name'], rawdata['Wavelength'], rawdata[type_of_data]):
            if n == name:
                # Read the data for the sample in question
                tmp_wlength.append(float(w))
                tmp_u_wlength.append(5.0)
                tmp_abs.append(float(a))
                if not uncertainty:
                    tmp_u_abs.append(DEFAULT_ABS_UNCERTAINTY * float(a))
        if uncertainty:
            for n, ua in zip(rawdata['Name'], rawdata['Uncertainty']):
                if n == name:
                    tmp_u_abs.append(float(ua))
        if mass_appo:
            for n, ec, oc in zip(rawdata['Name'], rawdata['EC'], rawdata['OC']):
                if n == name:
                    tmp_ec.append(float(ec))
                    tmp_oc.append(float(oc))
        for sample in data:
            if sample.name == name:
                # Populate the sample properties
                sample.properties.type_of_data = type_of_data #whether its ABS or b_abs
                sample.properties.wavelength = tmp_wlength
                sample.properties.u_wavelength = tmp_u_wlength
                sample.properties.abs = tmp_abs
                sample.properties.u_abs = tmp_u_abs
                if mass_appo:
                    tmp_ec = tmp_ec[0] # It's all the same values
                    tmp_oc = tmp_oc[0]
                    sample.properties.ec = tmp_ec
                    sample.properties.oc = tmp_oc
    return data

