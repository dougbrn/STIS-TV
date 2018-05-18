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
mpl.rcParams.update({'font.size': 18})
plt.rcParams["font.family"] = "Times New Roman"

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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.dpi = 200
    sigma = 5
    font_size = 12
    delta_clipped = sigma_clip(delta_scalevals, sigma=sigma)
    delta_clipped = np.ma.compressed(delta_clipped)

    delta_values, delta_base = np.histogram(delta_clipped, bins=40)
    delta_cumulative = np.cumsum(delta_values)
    percentile = delta_cumulative / len(delta_clipped) * 100
    p_vals = [50, 66, 75, 95]
    p_delta = np.interp(p_vals, percentile, delta_base[:-1], period=360)
    print(p_delta)

    # plot the cumulative function
    ax.plot(delta_base[:-1], percentile, c='k', label=" Delta Scale Values")
    ax.set_title("Change in Pixel Scale Value after an Anneal")
    ax.set_ylabel("Percentile")
    ax.set_xlabel(r"Delta Scale Value")
    ax.set_ylim(min(percentile), 100)
    ax.set_xlim(0, p_delta[-1])
    ax.grid()

    last_val = 0
    last_delta = min(percentile)
    for val, delta in zip(p_vals, p_delta):
        ax.axvline(delta, linestyle='--', color='k', alpha=0.5)
        p_mask = (percentile < val) * (percentile > last_val)
        unc_reg = np.append(delta_base[:-1][p_mask], delta)
        p_reg = np.append(percentile[p_mask], val)
        unc_reg = np.append(np.array([last_delta]), unc_reg)
        p_reg = np.append(np.array([last_val]), p_reg)

        ax.fill_between(unc_reg, 0, p_reg, alpha=0.2)
        ax.text(delta - 0.005 * max(delta_clipped), 30, '{}th Percentile'.format(val), horizontalalignment='center',
                verticalalignment='center', transform=ax.transData, rotation=90, fontsize=font_size)
        last_val = val
        last_delta = delta

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size + 2)

    plt.savefig("plots/delta_scaleval_paper.png")

    plt.show()
