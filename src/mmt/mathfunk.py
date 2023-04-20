#####################################################################
# IMPORTS
#####################################################################

import numpy as np


#####################################################################
# CONSTANT
#####################################################################




#####################################################################
# FUNCTION DEFINITIONS
#####################################################################

def singlefit(x, scale, aae):
    """The function for the overall power-law fit"""
    return scale * x ** ( - aae )


def doublefit(x, scale_1, alpha_1, scale_2, alpha_2):
    """The function for the double power_law fit"""
    return scale_1 * x ** ( - alpha_1) + scale_2 * x ** ( - alpha_2 )




