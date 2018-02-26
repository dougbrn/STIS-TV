#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
from astropy.io import fits

#Optimal Parameters found by least squares
for params in csv.reader(open('sv_params.csv', 'r'),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC):
    result_params = params

def compute_sv(params, dark, year):
    """Compute the temperature scale value for a given dark and decimal year

    The fitted values contained in `params` describe a skewed norm + linear distribution
    with time dependent parameters. Using these values, calculate the temperature scale value
    at a specific time (`year`) and a specific darkrate (`darkrate`)

    Parameters
    ----------
    params : list
        List of fitted parameters describing a time dependent skewed norm +
        linear distribution
    dark : list
        the data matrix of dark rates for a dark frame
    year : float
        Decimal year, should change to MJD soon.

    Returns
    -------
    sv_matrix
        A matrix of scalevalues corresponding to each pixel in the dark frame
    """
    #Messily read in parameters
    loc_slope1,loc_int1,loc_slope2,loc_int2,scale_slope1,scale_int1,scale_slope2,scale_int2,shape_slope1,shape_int1,shape_slope2,shape_int2,c,d=params #darkrate dependence and time dep parameters

    #Time dependent parameters for a skewed norm distribution

    loc = np.piecewise(year,
                           [(year <= 2010.3),(year > 2010.3)],
                           [lambda year:loc_slope1 * year + loc_int1,
                            lambda year: loc_slope2 * year + loc_int2])
    scale = np.piecewise(year,
                           [(year <= 2010.3),(year > 2010.3)],
                           [lambda year:scale_slope1 * year + scale_int1,
                            lambda year:scale_slope2 * year + scale_int2])
    shape = np.piecewise(year,
                           [(year <= 2010.3),(year > 2010.3)],
                           [lambda year:shape_slope1 * year + shape_int1,
                            lambda year:shape_slope2 * year + shape_int2])



    #Rate dependent distribution -- Skewed Norm + Linear + const
    dark_array = np.array(dark)
    t = (dark_array-loc) / scale
    sv_matrix = 2 / scale * stats.norm.pdf(t) * stats.norm.cdf(shape*t) + c*dark_array + d
    return sv_matrix

if __name__ == "main":

    pass
    #sv_matrix = compute_sv(result_params,np.log10(synth_dark),synth_date)
    #scaled_dark = np.array(synth_dark)*(1+(sv_matrix)*(synth_temp-ref_temp))
    #print(sv_matrix[0][0],np.log10(synth_dark[0][0]),scaled_dark[0][0])
