import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import csv
from scipy import stats
import datetime
import time

def compute_sv(param_path, darkrate, year):
    """Compute the scale value matrix for a given dark

    For a given dark, use fitted parameters for a time-dependent skewed normal
    distribution to find a pixel by pixel scale value (fractional change in
    dark rate per degree C change in temperature). Scale value is dependent on
    the pixels observed dark rate at a reference temperature of 18 deg C and on
    the date of the observation.

    Parameters
    ----------
    param_path : str
        path to a csv file containing the fitted parameters for the model scale
        value curve
    darkrate : np.array
        matrix of dark rates (counts * gain / exposure time)
    year : float
        decimal year of the observation

    Returns
    -------
    computed_sv : np.array
        matrix of scale values to be applied to the dark rate matrix
    """

    #Read in parameters from csv file
    with open(param_path,'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for readrow in reader:
            fit_params = np.array(readrow).astype(float)

    loc_slope1,loc_int1,loc_slope2,loc_int2,scale_slope1,scale_int1,scale_slope2,scale_int2,shape_slope1,shape_int1,shape_slope2,shape_int2,c,d=fit_params #darkrate dependence and time dep parameters
    dt_arr = year
    #Time dependent parameters
    loc = np.piecewise(dt_arr,
                           [(dt_arr < 2010.3),(dt_arr>2010.3)],
                           [lambda dt_arr:loc_slope1*dt_arr+loc_int1,
                            lambda dt_arr: loc_slope2*dt_arr+loc_int2])
    scale = np.piecewise(dt_arr,
                           [(dt_arr <= 2010.3),(dt_arr>2010.3)],
                           [lambda dt_arr:scale_slope1*dt_arr+scale_int1,
                            lambda dt_arr:scale_slope2*dt_arr+scale_int2])
    shape = np.piecewise(dt_arr,
                           [(dt_arr <= 2010.3),(dt_arr>2010.3)],
                           [lambda dt_arr:shape_slope1*dt_arr+shape_int1,
                            lambda dt_arr:shape_slope2*dt_arr+shape_int2])
    #Rate dependent factor
    t = (darkrate-loc) / scale
    computed_sv = 2 / scale * stats.norm.pdf(t) * stats.norm.cdf(shape*t) + c*darkrate + d

    return computed_sv

def calc_year(darkfile):
    """Calculate the decimal year of a dark

    Use the DATE-OBS header keyword to retrieve the decimal year form of a
    darks observation date.

    Parameters
    ----------
    darkfile : str
        name of a dark file

    Returns
    -------
    year : float
        decimal year of the darks observation date
    """
    dark = fits.open(darkfile)
    date = datetime.datetime(*[int(item) for item in dark[1].header['DATE-OBS'].split('-')])
    startOfThisYear = datetime.datetime(year=date.year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=date.year+1, month=1, day=1)
    yearElapsed = time.mktime(date.timetuple()) - time.mktime(startOfThisYear.timetuple())
    yearDuration = time.mktime(startOfNextYear.timetuple()) - time.mktime(startOfThisYear.timetuple())
    year = date.year + yearElapsed/yearDuration
    return year

if __name__ == "__main__":
    pass
