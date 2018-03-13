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
plt.style.use('seaborn-white')
mpl.rcParams.update({'font.size': 16})

# Select Dark
source_path = "/Users/dbranton/STIS/forks/refstis/refstis/tests/data/sources/"
refpath_list = ["/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/orig_firstorder/",
                "/Users/dbranton/STIS/forks/refstis/refstis/tests/data/products/diffref_firstorder/"]
order_list = ['First-Order', 'First-Order']
color_list = ['k', 'r']
dark = fits.open(os.path.join(source_path, "ocqq9tq6q_flt.fits"))

# Grab header info
ATODGAIN = dark[0].header['ATODGAIN']
TDATEOBS = dark[0].header['TDATEOBS']
EXPTIME = dark[1].header['EXPTIME']
OCCDHTAV = dark[1].header['OCCDHTAV']
DRK_VS_T = 0.07
dark_data = dark[1].data
dq_flag = (dark['DQ'].data != 16) & (dark['DQ'].data != 16384)
reffile_name = dark[0].header['DARKFILE'].split('$')[-1]


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

fig = plt.figure(figsize=(15, 10))
fig.dpi = 100
binwidth = 0.5
plt.title("Comparing Residual Distribution in {}".format(TDATEOBS[0:4]))
stats_text = ''

# Grab and scale dark reference files for comparison
for refpath, path_order, color in zip(refpath_list, order_list, color_list):
    reffile = fits.open(os.path.join(refpath, reffile_name))
    REF_TEMP = reffile[0].header['REF_TEMP']
    refdata = reffile[1].data * EXPTIME / ATODGAIN
    if path_order == "First-Order":
        refscaled = refdata * (1 + DRK_VS_T * (OCCDHTAV - REF_TEMP))
    elif path_order == "Second-Order":
        dark_svrates = np.copy(refdata)
        dark_svrates[dark_svrates <= 0] = 10 ** -3.0
        dark_svrates[dark_svrates >= 10 ** 1.0] = 10 ** 1.0
        sv_matrix = np.array(compute_sv(result_params, np.log10(dark_svrates), year))
        refdata = refdata * EXPTIME / ATODGAIN
        refscaled = refdata * (1 + (sv_matrix) * (OCCDHTAV - REF_TEMP))
    else:
        print("Not a supported Order")
        break

    sigma = 5
    mask = (dark_data / EXPTIME) > 0.0
    darkdiff = (dark_data - refscaled)
    masked = darkdiff[mask]

    sclipped = sigma_clip(masked, sigma=sigma)
    sclipped = np.ma.compressed(sclipped)

    plt.hist(np.ravel(sclipped), bins=np.arange(min(sclipped), max(sclipped) + binwidth, binwidth), alpha=0.7,
             label="{} : {}".format(path_order, REF_TEMP), color=color)

    stats_text = "{} ({}) Median: {}\n".format(path_order, REF_TEMP, str(np.round(np.median(np.ravel(sclipped)),
                                                                                  decimals=3)))
    stats_text += "{} ({}) Standard Deviation: {}\n".format(path_order, REF_TEMP, np.round(np.std(np.ravel(sclipped)),
                                                                                           decimals=3))
    print("{} ({}) Median: {}".format(path_order, REF_TEMP, np.median(np.ravel(sclipped))))
    print("{} ({}) Standard Deviation: {}".format(path_order, REF_TEMP, np.std(np.ravel(sclipped))))


plt.legend()
plt.grid()
plt.xlabel("Counts")
plt.ylabel("Frequency")
plt.show()
# plt.savefig("plots/correct_comp_2011_all.png")