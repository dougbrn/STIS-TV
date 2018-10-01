from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import csv
from astropy.stats import sigma_clip
from scipy import interpolate
import os
import datetime
from pixelcorrection import compute_sv
import matplotlib as mpl
import time
import glob
import json
plt.style.use('seaborn-paper')
mpl.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Times New Roman"

# Select Dark
source_2017 = "odbp4vyqq_flt.fits"
source_2011 = 'obn31dqgq_flt.fits'
source_path = "/Users/dbranton/STIS/forks/refstis/refstis/tests/data/weekdarks/"
#source_list = glob.glob(os.path.join(source_path, source_2017))
source_list = [os.path.join(source_path, source_2011), os.path.join(source_path, source_2017)]
"""
refpath_list = ["/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/orig_firstorder/",
                "/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/orig_secorder/",
                "/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/diffref_firstorder/"]
order_list = ['First-Order', 'Second-Order', 'First-Order']
color_list = ['k', 'r', 'b']

"""
refpath_list = ["/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/orig_firstorder/",
                "/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/orig_firstorder/"]
order_list = ['First-Order', 'Second-Order']
color_list = ['k', 'mediumpurple']

make_plot = True
write_fits = True

for idx, source in enumerate(source_list):

    print(source)
    dark = fits.open(source)
    # Grab header info
    ATODGAIN = dark[0].header['ATODGAIN']
    TDATEOBS = dark[0].header['TDATEOBS']
    EXPTIME = dark[1].header['EXPTIME']
    OCCDHTAV = dark[1].header['OCCDHTAV']
    print(OCCDHTAV)
    print(TDATEOBS)
    stats_text = ''
    DRK_VS_T = 0.07
    dark_data = dark[1].data
    dq_flag = (dark['DQ'].data != 16) & (dark['DQ'].data != 16384)
    reffile_name = dark[0].header['DARKFILE'].split('$')[-1]

    if make_plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        fig.dpi = 200
        binwidth = 20

    if "Second-Order" in order_list:
        # Read in second-order parameters
        for params in csv.reader(open('sv_params.csv', 'r'), delimiter=' ', quoting=csv.QUOTE_NONNUMERIC):
            result_params = params

        # Retreive decimal year, exposure length and darkrates for second-order method
        date = datetime.datetime(*[int(item) for item in TDATEOBS.split('-')])
        startOfThisYear = datetime.datetime(year=date.year, month=1, day=1)
        startOfNextYear = datetime.datetime(year=date.year + 1, month=1, day=1)
        yearElapsed = time.mktime(date.timetuple()) - time.mktime(startOfThisYear.timetuple())
        yearDuration = time.mktime(startOfNextYear.timetuple()) - time.mktime(startOfThisYear.timetuple())
        year = date.year + yearElapsed / yearDuration

    # Grab and scale dark reference files for comparison
    mu_list = []
    std_list = []
    resid_list = []
    lvl_list = []
    for hist_idx, contents in enumerate(zip(refpath_list, order_list, color_list)):
        refpath, path_order, color = contents
        reffile = fits.open(os.path.join(refpath, reffile_name))
        REF_TEMP = reffile[0].header['REF_TEMP']
        refdata = reffile[1].data

        if path_order == "First-Order":
            refdata = refdata * EXPTIME / ATODGAIN
            refscaled = refdata * (1 + DRK_VS_T * (OCCDHTAV - REF_TEMP))
        elif path_order == "Second-Order":
            dark_svrates = np.copy(dark_data) / EXPTIME * ATODGAIN  # Use values from dark (not ref) to scale temp
            dark_svrates = dark_svrates / (1 + 0.07 * (float(OCCDHTAV) - REF_TEMP))  # Approx ref temp dark rates
            dark_svrates[dark_svrates <= 10**-2.5] = 10 ** -2.5
            dark_svrates[dark_svrates >= 10 ** 1.0] = 10 ** 1.0
            #sv_matrix = np.array(compute_sv(result_params, np.log10(dark_svrates), year))


            scale_dict = json.load(open('scaleval_data.json'))
            gridpoints = np.array(np.meshgrid(scale_dict['Darkrate'],
                                              scale_dict['Datetimes'])).T.reshape(-1, 2)
            xlen, ylen = np.shape(dark_svrates)
            # gridvals = scalevalues.reshape(xlen*ylen,1)
            gridvals = np.ravel(scale_dict['Values'])
            range_points = np.array(np.meshgrid(np.ravel(np.log10(dark_svrates)), [year])).T.reshape(-1, 2)
            sv_matrix = interpolate.griddata(gridpoints, gridvals, range_points, method='linear').reshape(xlen, ylen)

            refdata = refdata * EXPTIME / ATODGAIN

            refscaled = refdata * (1 + sv_matrix * (OCCDHTAV - REF_TEMP))

        else:
            print("Not a supported Order")
            break

        sigma = 5

        # General
        # -----------------------------------
        mask = np.log10(dark_data / EXPTIME) < 1.0
        binwidth = 0.5
        # -----------------------------------
        # OR
        # Hot pixels
        # -----------------------------------
        #mask = np.log10(dark_data/EXPTIME) > 1.0
        #binwidth = 50
        # -----------------------------------

        darkdiff = (dark_data - refscaled)
        print("Uncorrected Median: ", np.median(np.ravel(dark_data)))
        print("Uncorrected Standard Deviation: ", np.std(np.ravel(dark_data)))
        mask = mask * dq_flag
        value_mask = (darkdiff > -3000) * (darkdiff < 1000)
        mask = mask * value_mask
        masked = darkdiff[mask]

        sclipped = sigma_clip(masked, sigma=sigma)
        sclipped = np.ma.compressed(sclipped)

        if make_plot:

            plt.hist(np.ravel(sclipped), bins=np.arange(min(sclipped), max(sclipped) + binwidth, binwidth),
                     alpha=0.8-hist_idx*0.2, color=color,
                     label=r"{} : $\mu$ = {},  $\sigma$ = {}".format(path_order,
                                                               np.round(np.median(np.ravel(sclipped)), decimals=3),
                                                               np.round(np.std(np.ravel(sclipped)), decimals=3)))

        mu_list.append(str(np.round(np.median(np.ravel(sclipped)), decimals=3)))
        std_list.append(str(np.round(np.std(np.ravel(sclipped)), decimals=3)))
        resid_list.append(darkdiff)
        lvl_list.append(sclipped)


        print("{} ({}) Median: {}".format(path_order, REF_TEMP, np.median(np.ravel(sclipped))))
        print("{} ({}) Standard Deviation: {}".format(path_order, REF_TEMP, np.std(np.ravel(sclipped))))

    if write_fits:
        for resid, order in zip(resid_list, order_list):
            hdu = fits.PrimaryHDU(abs(resid))
            hdu.writeto('{}.fits'.format(order), clobber = True)
    print(mu_list)
    print(lvl_list)
    print((abs(float(mu_list[1])) - abs(float(mu_list[0])))/abs(float(mu_list[0])) * 100)
    if make_plot:
        font_size = 14
        ax.set_title(r"Residual Distribution in {} (Temp: {} $\degree$C)".format(TDATEOBS[0:4], OCCDHTAV))
        ax.legend(fontsize=font_size)
        ax.grid()
        ax.set_xlabel("Residual Counts")
        ax.set_ylim(0, 30000)
        ax.set_ylabel("Frequency")
        """
        for i in range(len(mu_list)):
            ax.text(0.05, 0.8-0.03*i, r'{}: $\mu={},\ \sigma={}$'.format(order_list[i],
                                                                            mu_list[i],
                                                                            std_list[i]), transform=ax.transAxes,
                    fontsize=8)
        """
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size + 2)

        plt.savefig('Plots/resid_{}_paper.png'.format(TDATEOBS[0:4]))
        plt.show()


