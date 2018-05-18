import numpy as np
import pandas as pd
import glob
import datetime as dt
import json

"""This script stores all scalevalue results into a JSON file"""

imdir = "results_v3/" # results directory
lograte_regions = np.linspace(-2.5, 1, 15)
region_rates = [np.array([]) for _ in range(len(lograte_regions))]
region_errs = [np.array([]) for _ in range(len(lograte_regions))]
dt_mid = []

# Iterate over all files in results directory
for file in glob.glob(imdir + "*"):
    f_name = file.split("/")[-1]
    f_dates = f_name.split(".")[0]
    y_start, m_start, y_end, m_end = f_dates.split("_")

    dt_start = dt.datetime(int(y_start), int(m_start), 1)
    dt_end = dt.datetime(int(y_end), int(m_end), 1)
    delta = dt_end - dt_start
    dt_mid.append(dt_start + delta / 2)

    data = pd.read_csv(file, header=None)
    # Append rates and errors to their respective arrays
    for idx, row in enumerate(data.itertuples()):
        region_rates[idx] = np.append(region_rates[idx], row[2])
        region_errs[idx] = np.append(region_errs[idx], row[3])

dt_stamps = [mid.timestamp() / 3600 / 24 / 365 + 1970 for mid in dt_mid]

scalevalues = np.array(region_rates)
datetimes = np.array(dt_stamps)
darkrate_bins = np.array(lograte_regions)

# Store arrays into a dictionary
scale_dict = {'Values': scalevalues.tolist(),
              'Datetimes': datetimes.tolist(),
              'Darkrate': darkrate_bins.tolist()}
# Write to JSON file
with open('scaleval_data.json', 'w') as outfile:
    json.dump(scale_dict, outfile, ensure_ascii=False)
