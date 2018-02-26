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
mpl.rcParams.update({'font.size': 18})

def compute_scalevalue(pix):
    #Outlier rejection --  2 layers. Why? Extreme elevated pixels can mask
    #lower elevated outliers. First layer removes egregious outliers, second removes the rest
    #First layer of outlier rejection
    #import pdb; pdb.set_trace()
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
    if scale_value < 0:
        scale_value = 0
    #pix_scalevalues.append(scale_value)

    print("Scale Value: ",  scale_value)
    print("R-Squared: ", r_value**2)
    print("")
    return scale_value,r_value**2

if __name__ == "__main__":
    datadir = "/Users/dbranton/STIS/CCD-TV/pixelhist_data/"
    anneal_paths = [datadir+'pre_anneal',datadir+'post_anneal']

    cutout_size = 100

    anneal_scalevals = []
    anneal_rvals = []
    for path in anneal_paths:
        darks = glob.glob(path+'/*flt.fits')
        cutout_counts = []
        temps = []


        for dark in darks:
            dark_hdu = fits.open(dark)
            temp = dark_hdu[1].header['OCCDHTAV']
            exptime = dark_hdu[1].header['EXPTIME']

            cutout = dark_hdu[1].data[500:500+cutout_size,500:500+cutout_size]
            dq_cutout = dark_hdu['DQ'].data[500:500+cutout_size,500:500+cutout_size]

            #bit_mask = 4
            #dq_mask = np.bitwise_and(dq_cutout,np.zeros(np.shape(dq_cutout),'Int16')+bit_mask).astype('bool')
            #print(np.sum(dq_mask))
            #masked_cutout = cutout[~dq_mask]

            temps.append(temp)
            #print(np.shape(masked_cutout))
            #import pdb; pdb.set_trace()
            darkrates = [pixel for pixel in np.ravel(cutout)/exptime]
            cutout_counts.append(darkrates)

        #import pdb;pdb.set_trace()
        pix_darkrates = np.array(cutout_counts).T

        p = Pool(8)
        pix_svr = p.map(compute_scalevalue,pix_darkrates)
        anneal_scalevals.append(np.array([pix[0] for pix in pix_svr]))
        anneal_rvals.append(np.array([pix[1] for pix in pix_svr]))

    pre_scalevals = anneal_scalevals[0]
    post_scalevals = anneal_scalevals[1]
    pre_rvals = anneal_rvals[0]
    post_rvals = anneal_rvals[1]

    delta_scalevals = abs(pre_scalevals-post_scalevals)
    print(np.mean(delta_scalevals), np.std(delta_scalevals))

    total_pix = len(delta_scalevals)

    print("Percent of Pixels within 0.03: {}% ".format((len(delta_scalevals[np.where(abs(delta_scalevals) < 0.03)])/total_pix)*100))
    print("Percent of Pixels within 0.05: {}%".format((len(delta_scalevals[np.where(abs(delta_scalevals) < 0.05)])/total_pix)*100))
    print("Percent of Pixels within 0.10: {}%".format((len(delta_scalevals[np.where(abs(delta_scalevals) < 0.10)])/total_pix)*100))

    fig, ax = plt.subplots(1,1,figsize=(15,10))
    fig.dpi = 200
    sigma = 5
    delta_clipped = sigma_clip(delta_scalevals,sigma=sigma)
    delta_clipped = np.ma.compressed(delta_clipped)

    bins = np.histogram(delta_clipped,bins=50)[1]
    ax.hist(delta_clipped,histtype='bar',ec='black',bins=bins,alpha=0.7,color='b')
    ax.axvline(0.03,linestyle = '--',color = 'lightgreen',label = "Percent of Pixels within 0.03: {}% ".format((len(delta_scalevals[np.where(abs(delta_scalevals) < 0.03)])/total_pix)*100))
    ax.axvline(0.05,linestyle = '--',color = 'orange',label = "Percent of Pixels within 0.05: {}% ".format((len(delta_scalevals[np.where(abs(delta_scalevals) < 0.05)])/total_pix)*100))
    ax.axvline(0.10,linestyle = '--',color = 'red', label = "Percent of Pixels within 0.10: {}% ".format((len(delta_scalevals[np.where(abs(delta_scalevals) < 0.10)])/total_pix)*100))


    ax.set_xlim(-0.025,1.025)
    ax.set_title("Scale Value Anneal Difference for {}x{} Pixel Cutout".format(cutout_size,cutout_size))
    ax.set_ylabel("Frequency")
    ax.set_xlabel(" Delta Scale Value")
    plt.legend()
    plt.savefig("plots/delta_scaleval.png")
    plt.show()
