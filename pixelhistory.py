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
plt.rcParams["font.family"] = "Times New Roman"


def compute_scalevalue(pix):
    """Compute the scale value for a particular pixel."""

    # Outlier rejection --  2 layers. Why? Extreme elevated pixels can mask
    # lower elevated outliers. First layer removes egregious outliers, second removes the rest
    # First layer of outlier rejection
    regression = ols("data ~ x", data = dict(data=pix,x=temps)).fit()
    test = regression.outlier_test()
    outliers = ((temps[i],pix[i]) for i,t in enumerate(test['bonf(p)'].tolist()) if t < 0.5)
    outcorr_temps = [temps[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]
    outcorr_darkrates = [pix[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]

    # Second layer of outlier rejection
    regression = ols("data ~ x", data = dict(data=outcorr_darkrates,x=outcorr_temps)).fit()
    test = regression.outlier_test()
    outcorr_temps = [outcorr_temps[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]
    outcorr_darkrates = [outcorr_darkrates[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]

    slope,intercept,r_value,p_value,std_err = stats.linregress(outcorr_temps,outcorr_darkrates)
    scale_value = slope/(intercept + slope*18.0)
    #scale_value = slope
    sv_unc = std_err/(intercept + slope*18.0)
    #pix_scalevalues.append(scale_value)

    print("Scale Value: ",  scale_value)
    print("Scale Value Uncertainty: ", sv_unc)
    print("R-Squared: ", r_value**2)
    print("")
    return scale_value,sv_unc,r_value**2


if __name__ == "__main__":

    datadir = "/Users/dbranton/STIS/refstis/refstis/testdata/2013data/backup_flts/*flt.fits"

    cutout_size = 100
    darks = glob.glob(datadir)
    cutout_counts = []
    poisson_counts = []
    temps = []
    poisson_calc = True



    for dark in darks:
        dark_hdu = fits.open(dark)
        temp = dark_hdu[1].header['OCCDHTAV']
        exptime = dark_hdu[1].header['EXPTIME']
        cutout = dark_hdu[1].data[100:100+cutout_size,700:700+cutout_size]
        cutout_shape = np.shape(cutout)
        temps.append(temp)
        darkrates = [pixel for pixel in np.ravel(cutout)/exptime]
        cutout_counts.append(darkrates)
        poisson_noise = np.random.poisson(lam=18.7, size= cutout_size**2)/exptime
        poisson_counts.append(poisson_noise)

    pix_darkrates = np.array(cutout_counts).T
    pix_poisson = np.array(poisson_counts).astype(float).T +0.001



    p = Pool(8)
    pix_svr = p.map(compute_scalevalue,pix_darkrates)
    pix_scalevals = np.array([pix[0] for pix in pix_svr])
    pix_uncvals = np.array([pix[1] for pix in pix_svr])
    pix_rvals = np.array([pix[2] for pix in pix_svr])

    if poisson_calc == True:
        print("Poisson Scale Values")
        g = Pool(8)
        pix_p_svr = g.map(compute_scalevalue,pix_poisson)
        pix_p_scalevals = np.array([pix[0] for pix in pix_p_svr])
        pix_p_uncvals = np.array([pix[1] for pix in pix_p_svr])
        pix_p_rvals = np.array([pix[2] for pix in pix_p_svr])

        pix_p_scalevals = pix_p_scalevals + np.mean(pix_scalevals)

    total_pix = len(pix_scalevals)

    print("Percent of Pixels within 0.03 of First-Order: {}% ".format((len(pix_scalevals[np.where(abs(pix_scalevals - 0.07) < 0.03)])/total_pix)*100))
    print("Percent of Pixels within 0.05 of First-Order: {}%".format((len(pix_scalevals[np.where(abs(pix_scalevals - 0.07) < 0.05)])/total_pix)*100))
    print("Percent of Pixels within 0.10 of First-Order: {}%".format((len(pix_scalevals[np.where(abs(pix_scalevals - 0.07) < 0.10)])/total_pix)*100))

    fig, ax = plt.subplots(1,1,figsize=(10,6))
    fig.dpi = 200
    sigma = 5
    font_size = 12

    pix_clipped = sigma_clip(pix_scalevals,sigma=sigma)
    pix_clipped = np.ma.compressed(pix_clipped)
    print("Pixel Mean: ", np.mean(pix_clipped))
    print("Pixel Standard Deviation: ",np.std(pix_clipped))

    if poisson_calc == True:
        poisson_clipped = sigma_clip(pix_p_scalevals,sigma=sigma)
        poisson_clipped = np.ma.compressed(poisson_clipped)
        print("Poisson Mean: ", np.mean(poisson_clipped))
        print("Poisson Standard Deviation: ",np.std(poisson_clipped))

        poisson_frac = np.sqrt(np.std(pix_clipped)**2 - np.std(poisson_clipped)**2)/np.std(pix_clipped)
        print("Percentage of Spread Accounted for by poisson noise: ",(1-poisson_frac)*100)

    #Histogram comparison
    ax.grid()
    bins_poisson = np.histogram(poisson_clipped,bins=20)[1]
    ax.hist(poisson_clipped,histtype='bar',ec='black',bins=bins_poisson,alpha=0.7,color='k',label='Scale Value from Poisson Noise')

    bins = np.histogram(pix_clipped, bins=50)[1]
    ax.hist(pix_clipped, histtype='bar', ec='black', bins=bins, alpha=0.7, color='mediumpurple', label='Scale Value')
    #ax.axvline(0.07,color='k',label='First-Order Scale Value: 0.07')
    ax.set_title("Scale Value Distribution for {}x{} Pixel Cutout".format(cutout_size,cutout_size))
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Scale Value")

    ax.legend(fontsize=font_size)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size + 2)
    plt.savefig("Plots/pixeldist_vs_poissondist_paper.png")

    plt.show()
