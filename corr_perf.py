from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import csv
from astropy.stats import sigma_clip
import os
import datetime
from pixelcorrection import compute_sv
import matplotlib as mpl
import time
import glob
import operator
plt.style.use('seaborn-white')
mpl.rcParams.update({'font.size': 16})

# Select Dark
#source = "ocqq9tq6q_flt.fits"

weekdark_path = "/Users/dbranton/STIS/forks/refstis/refstis/tests/data/weekdarks/"
weekdark_list = glob.glob(os.path.join(weekdark_path, '*'))

refpath_list = ["/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/orig_firstorder/",
                "/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/orig_secorder/"]
order_list = ['First-Order', 'Second-Order']

ref_dict = {}
for idx, source in enumerate(weekdark_list):

    print(source)
    dark = fits.open(source)
    # Grab header info
    ATODGAIN = dark[0].header['ATODGAIN']
    TDATEOBS = dark[0].header['TDATEOBS']
    EXPTIME = dark[1].header['EXPTIME']
    OCCDHTAV = dark[1].header['OCCDHTAV']
    print(OCCDHTAV)
    print(TDATEOBS)
    DRK_VS_T = 0.07
    dark_data = dark[1].data
    dark_err = dark['ERR'].data
    dq_flag = (dark['DQ'].data != 16) & (dark['DQ'].data != 16384)
    reffile_name = dark[0].header['DARKFILE'].split('$')[-1]
    if reffile_name not in ref_dict.keys():
        ref_dict[reffile_name] = np.array([])

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
    results = []
    for hist_idx, contents in enumerate(zip(refpath_list, order_list)):
        refpath, path_order = contents
        reffile = fits.open(os.path.join(refpath, reffile_name))
        REF_TEMP = reffile[0].header['REF_TEMP']
        refdata = reffile[1].data
        referror = reffile['ERR'].data

        if path_order == "First-Order":
            refdata = refdata * EXPTIME / ATODGAIN
            referror = referror * EXPTIME / ATODGAIN
            refscaled = refdata * (1 + DRK_VS_T * (OCCDHTAV - REF_TEMP))
            referror = referror * (1 + DRK_VS_T * (OCCDHTAV - REF_TEMP))
        elif path_order == "Second-Order":
            dark_svrates = np.copy(dark_data) / EXPTIME * ATODGAIN  # Use values from dark (not ref) to scale temp
            dark_svrates = dark_svrates / (1 + 0.07 * (float(OCCDHTAV) - REF_TEMP))  # Approx ref temp dark rates
            dark_svrates[dark_svrates <= 0] = 10 ** -3.0
            #dark_svrates[dark_svrates >= 10 ** 2.0] = 10 ** 2.0

            sv_matrix = np.array(compute_sv(result_params, np.log10(dark_svrates), year))
            refdata = refdata * EXPTIME / ATODGAIN
            referror = referror * EXPTIME / ATODGAIN
            refscaled = refdata * (1 + (sv_matrix) * (OCCDHTAV - REF_TEMP))
            referror = referror * (1 + (sv_matrix) * (OCCDHTAV - REF_TEMP))
        else:
            print("Not a supported Order")
            break

        # Create slit cutouts
        n_row = 100
        slit_width = 4

        dark_cutout = dark_data[n_row:n_row + slit_width]
        ref_cutout = refscaled[n_row:n_row + slit_width]

        dark_err_cutout = dark_err[n_row:n_row + slit_width]
        ref_err_cutout = referror[n_row:n_row + slit_width]


        # Mask out hot pixels
        sigma = 5
        dark_sclip = sigma_clip(dark_cutout, sigma=sigma)
        dark_mask = np.ravel(dark_sclip.mask)

        ref_sclip = sigma_clip(ref_cutout, sigma=sigma)
        ref_mask = np.ravel(ref_sclip.mask)

        mask = ref_mask * dark_mask

        #import pdb;pdb.set_trace()

        # Get percent difference -- maybe instead of summing these, take residual first and do pdiff of each pixel?

        dark_sum = np.sum(np.ravel(dark_cutout)[~mask])
        ref_sum = np.sum(np.ravel(ref_cutout)[~mask])

        dark_err_masked = np.ravel(dark_err_cutout)[~mask]
        ref_err_masked = np.ravel(ref_err_cutout)[~mask]

        dark_cut_err = np.sqrt(sum(map(lambda x: x * x, dark_err_masked)))
        ref_cut_err = np.sqrt(sum(map(lambda x: x * x, ref_err_masked)))

        pdiff = ((ref_sum - dark_sum) / dark_sum) * 100
        err_term = np.sqrt(ref_cut_err**2 + dark_cut_err**2)
        pdiff_err = abs(pdiff) * np.sqrt(
            (err_term / (ref_sum - dark_sum)) ** 2 + (dark_cut_err / dark_sum) ** 2)

        results.append([pdiff, pdiff_err])

    # Read in results of each method percent difference
    res_array = np.array(results).T
    delta_pdiff = abs(res_array[0][0]) - (abs(res_array[0][1]))  # Get the delta percent difference
    delta_pdiff_err = np.sqrt(res_array[1][0] ** 2 + res_array[1][1] ** 2)  # Get the delta percent difference error

    # Store pdiff and error in dictionary with key corresponding to the reference file used
    print(year)
    if len(ref_dict[reffile_name]) == 0:
        ref_dict[reffile_name] = [delta_pdiff, delta_pdiff_err, OCCDHTAV, year]
    else:
        ref_dict[reffile_name] = np.vstack([ref_dict.get(reffile_name), [delta_pdiff, delta_pdiff_err, OCCDHTAV, year]])

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
fig.dpi = 100
keys = list(ref_dict.keys())

pdiff_all = []
temp_all = []
err_all = []

for key in keys:
    pdiff_arr = ref_dict[key].T[0]
    err_arr = ref_dict[key].T[1]
    temp_arr = ref_dict[key].T[2]
    year_label = np.median(ref_dict[key].T[3])
    print(year)
    print(pdiff_arr)
    print(err_arr)
    color = plt.cm.cool((year_label - 2009)/(2021-2009))
    ax.errorbar(temp_arr, pdiff_arr, err_arr, label=str(np.round(year_label, 3)), fmt='.', capsize=2, color=color)
    pdiff_all.extend(pdiff_arr)
    temp_all.extend(temp_arr)
    err_all.extend(err_arr)

linear_fit = np.polyfit(temp_all,pdiff_all,1)
p = np.poly1d(linear_fit)
ax.plot(temp_all,p(temp_all),color='k')
ax.set_ylim(-4, 4)
ax.set_xlim(17.75, 24.25)
ax.set_ylabel("Percent Improvement on Dark Residuals for a {}x1024 Slit".format(slit_width))
ax.set_xlabel(r"CCD Housing Temperature ($\degree$C)")
ax.grid()
ax.set_title("Performance Comparison of First-Order and Second-Order Temperature Correction")
ax.fill_between([15, 25], -5, 0, alpha=0.3, color='salmon')
ax.fill_between([15, 25], 0, 5, alpha=0.2, color='slateblue')
ax.text(0.5, 0.9, 'Second-Order', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes,
        color='k')
ax.text(0.5, 0.1, 'First-Order', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes,
        color='k')
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels),
            key=operator.itemgetter(1))
handles2, labels2 = zip(*hl)

ax.legend(handles2, labels2)

plt.show()





