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
plt.style.use('seaborn-white')
mpl.rcParams.update({'font.size': 16})

def compute_scalevalue(pix):
    """Compute the scale value for a particular pixel."""

    #Outlier rejection --  2 layers. Why? Extreme elevated pixels can mask
    #lower elevated outliers. First layer removes egregious outliers, second removes the rest
    #First layer of outlier rejection
    regression = ols("data ~ x", data = dict(data=pix,x=temps)).fit()
    test = regression.outlier_test()
    outliers = ((temps[i],pix[i]) for i,t in enumerate(test['bonf(p)'].tolist()) if t < 0.5)
    outcorr_temps = [temps[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]
    outcorr_darkrates = [pix[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]

    #Second layer of outlier rejection
    regression = ols("data ~ x", data = dict(data=outcorr_darkrates,x=outcorr_temps)).fit()
    test = regression.outlier_test()
    outcorr_temps = [outcorr_temps[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]
    outcorr_darkrates = [outcorr_darkrates[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]

    slope,intercept,r_value,p_value,std_err = stats.linregress(outcorr_temps,outcorr_darkrates)
    scale_value = slope/(intercept + slope*18.0)
    sv_unc = std_err/(intercept + slope*18.0)

    print("Scale Value: ",  scale_value)
    print("Scale Value Uncertainty: ", sv_unc)
    print("R-Squared: ", r_value**2)
    print("")
    return scale_value,sv_unc,r_value**2

if __name__ == "__main__":

    datadir = "/Users/dbranton/STIS/refstis/refstis/testdata/2013data/backup_flts/*flt.fits"

    darks = glob.glob(datadir)
    cutout_counts = []
    temps = []

    for dark in darks:
        dark_hdu = fits.open(dark)
        temp = dark_hdu[1].header['OCCDHTAV']
        exptime = dark_hdu[1].header['EXPTIME']
        cutout = dark_hdu[1].data[500:520,:]
        cutout_shape = np.shape(cutout)
        temps.append(temp)
        darkrates = [pixel for pixel in np.ravel(cutout)/exptime]
        cutout_counts.append(darkrates)

    pix_darkrates = np.array(cutout_counts).T

    p = Pool(8)
    pix_svr = p.map(compute_scalevalue,pix_darkrates)
    pix_scalevals = np.array([pix[0] for pix in pix_svr])
    pix_uncvals = np.array([pix[1] for pix in pix_svr])
    pix_rvals = np.array([pix[2] for pix in pix_svr])


    fig, ax = plt.subplots(1,1,figsize=(15,10))
    fig.dpi = 200
    sigma = 5


    #2dhist -- Structure in scale values?
    pix_sv2d = np.reshape(pix_scalevals,cutout_shape).T

    #import pdb;pdb.set_trace()
    idx_list = []
    row_sv = []
    for idx,row in enumerate(pix_sv2d):
        idx_list.append(idx)
        row_sv.append(np.mean(row))
    #ax.imshow(pix_sv2d,cmap = 'viridis',vmin=0,vmax=0.4)

    slope,intercept,r_value,p_value,std_err = stats.linregress(idx_list,row_sv)

    print(slope,std_err,r_value)
    print("Slope: ",slope)
    print("Standard Error: ",std_err)
    print("R-Squared: ",r_value**2)
    linfit = intercept+np.array(idx_list)*slope

    ax.plot(idx_list,row_sv,'.',color='k')
    ax.plot(idx_list,linfit,color='r')
    ax.set_ylim(0.0,0.3)
    ax.set_xlabel("Column")
    ax.set_ylabel("Mean Scale Value")
    ax.set_title("Geometric Temp. Sensitivity: Detector X")
    ax.grid()

    #import pdb;pdb.set_trace()
    #plt.savefig("plots/detector_x.png")
    plt.show()
