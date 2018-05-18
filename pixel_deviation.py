#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import statsmodels.api as smapi
import statsmodels.graphics as smgraphics
from statsmodels.formula.api import ols
import pandas as pd
from scipy import stats
from astropy.stats import sigma_clip
from multiprocessing import Pool
import matplotlib as mpl
plt.style.use('seaborn-paper')
mpl.rcParams.update({'font.size': 14})

datadir = "/Users/dbranton/STIS/refstis/refstis/testdata/2013data/backup_flts/*flt.fits"
darks = glob.glob(datadir)
avg_counts = 18.7 #Counts, not dark rate
sigma = 5
fig, ax = plt.subplots(1,1,figsize=(15,10))
fig.dpi = 200
ref_temp = 18.0
avg_poisson = []
temps = []

def avg_counts_at_reftemp(temp):
    """Scale avg counts by first order approx"""
    scaled_avg = avg_counts*(1+0.07*(temp-ref_temp))
    return scaled_avg

for dark in darks[:]:
    dark_hdu = fits.open(dark)
    temp = dark_hdu[1].header['OCCDHTAV']
    exptime = dark_hdu[1].header['EXPTIME']
    scaled_avg = avg_counts_at_reftemp(temp)
    poisson_dist = np.random.poisson(lam=scaled_avg, size= 10000)
    poisson = np.std(poisson_dist)

    pixel_deviation = abs(np.array(dark_hdu[1].data) - scaled_avg)/poisson

    pixdev_clipped = sigma_clip(pixel_deviation,sigma=sigma)
    pixdev_clipped = np.ma.compressed(pixdev_clipped)

    avg_poisson.append(np.median(pixdev_clipped))
    temps.append(temp)

    #Histogram
    """
    bins = np.histogram(pixdev_clipped,bins=50)[1]
    ax.hist(pixdev_clipped,histtype='bar',ec='black',bins=bins,
            alpha=0.7, label = r'Temp: {} $\degree C$'.format(temp))
    """

"""
ax.set_title("Pixel Poisson Deviation from Mean Counts ")
ax.set_ylabel("Frequency")
ax.set_xlabel("n-Poisson")
"""

ax.plot(temps,avg_poisson,'.',color='k')
ax.set_title("Average Pixel Poisson Deviation vs. Temp. ")
ax.set_ylabel("n-Poisson")
ax.set_xlabel(r"Temperature $\degree C$")
ax.grid()
plt.legend()
#plt.savefig("Plots/pixel_deviation.png")
plt.show()
