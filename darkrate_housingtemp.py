#!/usr/bin/env python
import glob
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import datetime
import operator
import stistools as st
from itertools import zip_longest                   # for Python 3.x
from collections import Counter
from sklearn.preprocessing import Imputer
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from astropy.stats import sigma_clip
import yaml
import warnings
import time
from cycler import cycler
import csv
from multiprocessing import Pool, Manager, Process
import statsmodels.api as smapi
from statsmodels.formula.api import ols
import statsmodels.graphics as smgraphics

# ------------------------------------------------------------------------------

def retrieve_data(dataset,image_dir = '.'):
    """Retrieve a set of darks (flt files) from a specified directory

    Retrieve darks from path `image_dir`, specify files in `dataset` (i.e. *_flt.fits).
    Generates a dateobs list and returns the first and last dark dateobs values.

    Parameters
    ----------
    dataset : string
        Unix-type file pointer for all desired files in `image_dir`
    image_dir : string, optional
        directory of `dataset`, default value is '.'
    date_start : datetime.datetime(), optional
        Start date for optional date range specification. datetime object supports
        time as well. Default value is None, where no filter will be applied
    date_end : datetime.datetime(), optional
        End date for optional date range specification. datetime object supports
        time as well. Default value is None, where no filter will be applied

    Returns
    -------
    image_subset
        Subset of images (names) from `dataset`
    dateobs_subset
        Subset of image datetimes from `dataset`
    first_dark
        Datetime of the first dark from `dataset`
    last_dark
        Datetime of the last dark from `dataset`
    """
    images = glob.glob(image_dir+dataset)

    # Subset of images with known housing temperatures and in the appropriate date range
    image_subset=[]
    dateobs_subset = []
    for image in images:
        # Ensure a valid header temperature value and 1x1 binning
        if fits.getheader(image,1)['OCCDHTAV'] != -1.0 and fits.getheader(image)['BINAXIS1'] == 1:
            # retrieve datetime
            yyyy,mo,dd = fits.getheader(image)['TDATEOBS'].split('-')
            hh,mn,ss = fits.getheader(image)['TTIMEOBS'].split(':')
            dateobs = datetime.datetime(int(yyyy),int(mo),int(dd),int(hh),int(mn),int(ss))
            # Append to lists
            image_subset.append(image)
            dateobs_subset.append(dateobs)

    first_dark = min(dateobs_subset)
    last_dark = max(dateobs_subset)
    return image_subset, dateobs_subset, first_dark, last_dark

# ------------------------------------------------------------------------------


def anneal_split(image_subset,dateobs_subset,date_start,date_end,boundaries):
    """Split darks into separate anneal periods.

    For all darks in `subset`, split by keywords TTIMEOBS and TDATEOBS into
    separate anneal periods, generates anneal history up until July 2017 and
    splits by `date_start` and `date_end`.

    Note: These should probably be removed and just found from `dateobs_subset`

    Parameters
    ----------
    image_subset : list
        List of darks by string name
    dateobs_subset : list
        List of darks by date of observation
    date_start : datetime.datetime()
        Start date for date range specification. datetime object supports
        time as well.
    date_end : datetime.datetime()
        End date for date range specification. datetime object supports
        time as well.
    boundaries : list
        List of anneal boundaries.

    Returns
    -------
    ann_list_tsorted
        List containing darks in separate anneal period lists. Sorted by
        temperature and stored as tuples (filename, temperature)
    """
    #Generate list of anneal period start dates
    ann_hist = []
    for region in range(len(boundaries)):
        ann_hist.append(boundaries[region]['start'])

    anneals = np.array(ann_hist)[(np.array(ann_hist) < date_end) &
     (np.array(ann_hist) > date_start)]

    n_periods = len(anneals) + 1
    print("--> {} Anneal Periods defined.".format(n_periods))

    # Split Subset into groups based on Anneal History
    ann_list = [[] for _ in range(n_periods)]

    for image, im_datetime in zip(image_subset,dateobs_subset):

        # Find nearest anneal period
        nearest_ann = min(anneals, key = lambda x:abs(x-im_datetime))
        ann_idx, = np.where(anneals == nearest_ann)[0]
        ann_idx += 1 # If darks_before == True, which is now always the case

        # Determine whether the dark occured before or after the nearest anneal
        if im_datetime-nearest_ann > datetime.timedelta(0):
            ann_list[ann_idx].append(image)
        else:
            ann_list[ann_idx-1].append(image)

    for idx,ann_per in enumerate(ann_list):
        print("--> Anneal Period "+str(idx+1)+" :", "{} Darks".format(len(ann_list[idx])))

    #Sort each Anneal Period by temperature
    #Note that this returns lists of tuples not just lists of filenames
    ann_list_tsorted = [[] for _ in range(len(ann_list))]
    for idx,ann in enumerate(ann_list):
        ann_dict={}
        for file in ann:
            ann_dict[file] = fits.getheader(file,1)['OCCDHTAV']
            ann_list_tsorted[idx] = sorted(ann_dict.items(),key=operator.itemgetter(1))

    return ann_list_tsorted

# ------------------------------------------------------------------------------

def telem_temp(ann_list):
    """Generate housing temperatures for each dark from the telemetry file, OCCDHT

    For all darks in each anneal period in `ann_list`, calculate a housing
    temperature using the OCCDHT telemetry file. Fit nearby OCCDHT measurements
    and average over the exposure time.

    Parameters
    ----------
    ann_list : list
        List of anneal period lists


    Returns
    -------
    df_ann_list
        List containing anneal period lists with darks assigned housing temperatures
    """
    #Read in the telemetry file for the housing temperature
    hdrs = ['MJD','temp']
    telem = pd.read_csv('OCCDHT',delim_whitespace=True,header=None, names=hdrs)

    #Create arrays out of dataframe that are maskable
    telem_mjd = np.array(telem['MJD'])
    telem_temp = np.array(telem['temp'])


    df_ann_list = [None for _ in range(len(ann_list))]

    #Iterate over each anneal period
    for idx,ann in enumerate(ann_list):
        darks = []
        t_hdr = []
        t_avgs = []
        diff = []

    #Iterate over each (file,temp) tuple in the anneal period
        for dark in ann:
            image = dark[0]
            mjd_start = fits.getheader(image)['TEXPSTRT']
            mjd_end = fits.getheader(image)['TEXPEND']
            hdr_temp = fits.getheader(image,1)['OCCDHTAV']
            exp_mask = (telem_mjd <= mjd_end) & (telem_mjd >= mjd_start)
            mjd_exp_subset = telem_mjd[exp_mask]
            t_exp_subset = telem_temp[exp_mask]

            # If there are measurements made inside the MJD range of the exposure
            # itself, average those for the temp.
            if len(mjd_exp_subset) != 0:
                t_avg = np.sum(t_exp_subset)/len(t_exp_subset)

            # If not, create a polyfit of nearby points and average the
            # fitted curve over the exposure length.
            else:
                mjd_mask = (telem_mjd <= mjd_end + 0.1) & (telem_mjd >= mjd_start - 0.1)
                mjd_subset = telem_mjd[mjd_mask]
                temp_subset = telem_temp[mjd_mask]
                if len(mjd_subset) == 0:
                    print("--> No telemetry measurements made near {}".format(image))
                    continue
                with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        try:
                            temp_fit = np.polyfit(mjd_subset,temp_subset,2)
                            p = np.poly1d(temp_fit)
                        except np.RankWarning:
                            print("--> {} poorly fit, excluding from dataset.".format(image))
                            continue

                n_bins = 15
                exp_linspace = np.linspace(mjd_start,mjd_end,n_bins)
                t_avg = np.sum(p(exp_linspace))/n_bins

            #Store these calculations into lists
            darks.append(image)
            t_hdr.append(hdr_temp)
            t_avgs.append(t_avg)
            diff.append(t_avg-hdr_temp)

        #Store into anneal dataframes
        d = {'Dark': darks, 'Header Temp': t_hdr, 'Telem Temp':t_avgs, 'Difference':diff}
        cols = ['Dark','Header Temp', 'Telem Temp', 'Difference']
        df_ann_list[idx]=pd.DataFrame(data=d)[cols]


    #Sort dataframes by temperature
    for idx,ann in enumerate(df_ann_list):
        df_ann_list[idx] = ann.sort_values('Telem Temp')

    return df_ann_list

# ------------------------------------------------------------------------------

def cluster_crj(df_ann_list):
    """Cluster darks into triplets with roughly identical temperatures.

    For all darks in each anneal period in `df_ann_list`, cluster into triplets
    using a K-means algorithm. Fit nearby OCCDHT measurements
    and average over the exposure time.

    Parameters
    ----------
    df_ann_list : list
        List containing anneal period lists with darks assigned housing temperatures


    Returns
    -------
    clust_ann_list
        List containing anneal period lists with darks clustered into (rough) triplets
    """
    np.random.seed((1000,2000))

    # Cluster each anneal period by similar temperatures via a kmeans approach
    ann_temp_arrays = [[] for _ in range(len(df_ann_list))]
    ann_file_arrays = [[] for _ in range(len(df_ann_list))]

    for idx,df in enumerate(df_ann_list):
        ann_temp_arrays[idx] = np.array(df['Telem Temp'].tolist())
        ann_file_arrays[idx] = np.array(df['Dark'].tolist())

    darks_per_cluster = 3          #Want to generate triplets
    clust_ann_list = [
    [[] for _ in range(int(np.round(len(ann_temp_array)/darks_per_cluster)))]
    for ann_temp_array in ann_temp_arrays]



    for (ann_temp_array, ann_file_array,clust_ann) in zip(ann_temp_arrays,
                                                            ann_file_arrays,
                                                            clust_ann_list):
        codebook, _ = kmeans(ann_temp_array, int(np.round(len(ann_temp_array)/darks_per_cluster)))
        cluster_indices, _ = vq(ann_temp_array, codebook)

        for (value, file, idx) in zip(ann_temp_array,ann_file_array, cluster_indices):
            clust_ann[idx].append((file,value))


    return clust_ann_list

# ------------------------------------------------------------------------------

def ocrreject_clusters(clust_ann_list, outdir='.', hist_file = "combine_history.txt"):
    """Perform cosmic ray rejection on clustered darks.

    For all clustered darks in each anneal period in `clust_ann_list`, perform
    cosmic ray rejection on isothermal triplets to a specified output directory.
    Additionally, a file detailing the combination history is written out.

    Parameters
    ----------
    clust_ann_list : list
        List containing anneal period lists with darks clustered by temperature
    out_dir : string , optional
        Path to the desired crj file output directory. Default is '.'
    hist_file : string, optional
        Name of combination history file. Default is "combine_history.txt"

    Returns
    -------
    ann_labels
        List of labels for generated anneal periods

    """
    ann_labels = []
    f_out = open(hist_file,"w")

    for idx,clustered_ann in enumerate(clust_ann_list):
        ann_label = 'ann'+str(idx+1)
        if ann_label not in ann_labels:
            ann_labels.append(ann_label)
        f_out.write ("----------------------\n")
        f_out.write(ann_label+"\n")
        f_out.write ("----------------------\n")

        #Iterate over each cluster in the anneal period
        for cluster in clustered_ann:
            if cluster != 0:
                infile_list = [file[0] for file in cluster] #Just the files...
                temp_list = [file[1] for file in cluster]   #Just the temperatures...
                n_darks = len(infile_list)

                #Doublets are used
                if n_darks == 2:
                    print(infile_list)
                    doublet_list = infile_list
                    doublet_temp = temp_list
                    infile_str = ','.join(doublet_list)
                    avg_t = np.mean(doublet_temp)
                    outname = out_dir+str(avg_t)+'_'+str(1)+'_'+ann_label+'_crj.fits'
                    st.ocrreject.ocrreject(infile_str,outname,all=True)
                    hdr_temps = [fits.getheader(file,1)['OCCDHTAV'] for file in doublet_list]
                    f_out.write(infile_str+" --> "+outname+"\n")
                    f_out.write("Telemetry Temperatures: "+','.join(list(map(str,doublet_temp)))+"\n")
                    f_out.write("Header Temperatures: "+','.join(list(map(str,hdr_temps)))+"\n")
                    print("--> Doublet Created")

                # Triplets are created from clusters with 3 or more darks
                if n_darks >=3:
                    triplet_list = list(zip_longest(*[iter(infile_list)]*3, fillvalue=None)) #Split into triplet groups
                    triplet_temp = list(zip_longest(*[iter(temp_list)]*3, fillvalue=None))

                    #If n_darks is cleanly divisible by 3, divide into threes and ocrreject
                    if n_darks % 3 == 0: #3,6,...
                        for num in range(len(triplet_list)):
                            if num == 0 or num == 1 or num == 2:
                                infile_str = ','.join(triplet_list[num])
                                avg_t = np.mean(triplet_temp[num])
                                outname = out_dir+str(avg_t)+'_'+str(num+1)+'_'+ann_label+'_crj.fits'
                                st.ocrreject.ocrreject(infile_str,outname,all=True)
                                hdr_temps = [fits.getheader(file,1)['OCCDHTAV'] for file in triplet_list[num]]
                                f_out.write(infile_str+" --> "+outname+"\n")
                                f_out.write("Telemetry Temperatures: "+','.join(list(map(str,triplet_temp[num])))+"\n")
                                f_out.write("Header Temperatures: "+','.join(list(map(str,hdr_temps)))+"\n")
                            else:
                                print ("Skipped some darks for redundance rejection")

                    #If not, ignore any remaining darks
                    elif n_darks % 3 == 1: #4,7,...
                        for num in range(len(triplet_list)):
                            if None not in triplet_list[num]:
                                infile_str = ','.join(triplet_list[num])
                                avg_t = np.mean(triplet_temp[num])
                                outname = out_dir+str(avg_t)+'_'+str(num+1)+'_'+ann_label+'_crj.fits'
                                st.ocrreject.ocrreject(infile_str,outname,all=True)
                                hdr_temps = [fits.getheader(file,1)['OCCDHTAV'] for file in triplet_list[num]]
                                f_out.write(infile_str+" --> "+outname+"\n")
                                f_out.write("Telemetry Temperatures: "+','.join(list(map(str,triplet_temp[num])))+"\n")
                                f_out.write("Header Temperatures: "+','.join(list(map(str,hdr_temps)))+"\n")

                    #If not, ignore any remaining darks
                    elif n_darks % 3 == 2: #5,8,...
                        for num in range(len(triplet_list)):
                            if None not in triplet_list[num]:
                                infile_str = ','.join(triplet_list[num])
                                avg_t = np.mean(triplet_temp[num])
                                outname = out_dir+str(avg_t)+'_'+str(num+1)+'_'+ann_label+'_crj.fits'
                                st.ocrreject.ocrreject(infile_str,outname,all=True)
                                hdr_temps = [fits.getheader(file,1)['OCCDHTAV'] for file in triplet_list[num]]
                                f_out.write(infile_str+" --> "+outname+"\n")
                                f_out.write("Telemetry Temperatures: "+','.join(list(map(str,triplet_temp[num])))+"\n")
                                f_out.write("Header Temperatures: "+','.join(list(map(str,hdr_temps)))+"\n")

    f_out.close()
    return ann_labels

# ------------------------------------------------------------------------------

def choose_RF(ann_labels,out_dir='.'):
    """Choose Reference Frames for each anneal period

    Glob in all crj files from each anneal period in `ann_labels` in the output
    directory `out_dir`. Retreive each frames dark rate matrix (data/exptime) and
    store in the allframes dictionary. Additionally, search each anneal period
    for the two closest temperature darks to set as Reference Frames 1 and 2.

    Parameters
    ----------
    ann_labels : list
        List containing anneal period labels for glob parameter
    out_dir : string , optional
        Path to the desired crj file output directory. Default is '.'

    Returns
    -------
    allframes
        Dictionary containing each frame in every anneal period and the associated
        dark rate matrix
    RF_list
        List containing tuples with (RF1,RF2) where the index corresponds to each
        anneal period. Index + 1 = Anneal Period
    RF_temps
        List containing tuples with (RF1_temp,RF2_temp) where the index corresponds to each
        anneal period. Index + 1 = Anneal Period
    skip_idx:
        List of skipped anneal indexes

    """
    #Get crj files from out_dir
    frame_list=[[] for _ in range(len(ann_labels))]
    for idx,ann_label in enumerate(ann_labels):
        frame_list[idx] = glob.glob('{}*{}*'.format(out_dir,ann_label))

    allframes = {}
    RF_list=[]
    RF_temps=[]
    skip_idx = []

    for idx,ann in enumerate(frame_list):
        if len(ann) < 4:
            print("--> Anneal Period Skipped: Not enough frames")
            skip_idx.append(idx)
            continue
        ann_list = []
        for frame in ann:

            #Add Frame to allframes dictionary
            frame_hdr = fits.open(frame)
            exptime = float(frame_hdr['SCI'].header['EXPTIME'])
            frame_data = frame_hdr['SCI'].data
            allframes[frame] = np.ravel(frame_data)/exptime

            #Add Frame to anneal list for RF selection
            if "/" in frame:
                temp = float(frame.split("/")[-1].split("_")[0])
            else:
                temp = float(frame.split("_")[0])
            ann_list.append((frame,temp))
        ann_srtd = sorted(ann_list, key=lambda x: x[1])
        res = [ann_srtd[i + 1][1] - ann_srtd[i][1] for i in range(len(ann_srtd)) if i+1 < len(ann_srtd)]
        index = res.index(min(res))
        RF1 = ann_srtd[index][0]
        RF2 = ann_srtd[index+1][0]
        RF1_temp = ann_srtd[index][1]
        RF2_temp = ann_srtd[index+1][1]
        RF_list.append((RF1,RF2))
        RF_temps.append((RF1_temp,RF2_temp))
    return allframes, RF_list, RF_temps,skip_idx

# ------------------------------------------------------------------------------

def frame_ratio_calc(frame):
    """Calculate dark rate ratio for a frame against it's reference frames

    For all dark frames, find the corresponding RF1 and RF2. Take the ratio of
    the frame over RF1 and take the log (base 10) of RF2. Mask out nan values
    and ratios below 0. Calculate a running median over across RF2.

    Note: Not all parameters passed into function currently, only works in this pipe.
    Will get back to this and figure out a good way to pass these parameters in for
    multiprocessing

    Parameters
    ----------
    frame : str
        name of dark frame

    Returns
    -------
    frame
        name of dark frame
    temp
        temperature of dark frame
    running_median
        calculated ratio medians across log darkrate space
    avg_darkrate
        RF2 average darkrate for the dark frame

    """

    print("--> {}".format(frame))
    #Don't compare reference frames against themselves
    if len([f for f in RF_list if f[0] == frame]) > 0 or len([f for f in RF_list if f[1] == frame]) > 0:
        return
    f_name = frame.split('/')[-1]
    temp,num,ann,ext = f_name.split('_')

    ann_idx = int(ann[-1])-1
    for idx in skip_idx:
        if ann_idx >= idx:
            ann_idx-=1

    #dq_flags
    frame_flag = (np.ravel(fits.open(frame)['DQ'].data) !=16) & (np.ravel(fits.open(frame)['DQ'].data) != 256)
    RF1_flag = (np.ravel(fits.open(RF_list[ann_idx][0])['DQ'].data) !=16) & (np.ravel(fits.open(RF_list[ann_idx][0])['DQ'].data) != 256)
    dq_flag = np.array(frame_flag)*np.array(RF1_flag)
    frame_data = allframes.get(frame)[dq_flag]

    RF1 = allframes.get(RF_list[ann_idx][0])[dq_flag]
    RF2 = allframes.get(RF_list[ann_idx][1])[dq_flag]

    #Remove negative rates from RF2
    RF2_pos = RF2 > 0

    ratio = (frame_data / RF1)[RF2_pos]
    log_RF2 = np.log10(RF2[RF2_pos])

    #Remove Nan Values
    log_RF2_nonan = log_RF2[~np.isnan(log_RF2)]
    ratio_nonan = ratio[~np.isnan(log_RF2)]

    #Remove Negative Ratios
    ratio_mask = ratio_nonan >=0
    log_RF2_masked = log_RF2_nonan[ratio_mask]
    ratio_masked = ratio_nonan[ratio_mask]

    #Sigma Clip
    ratio_masked_sclip = sigma_clip(ratio_masked,sigma=3)
    ratio_masked_sclip = np.ma.compressed(ratio_masked_sclip)
    sclip_mask = np.isin(ratio_masked,ratio_masked_sclip)
    log_RF2_masked_sclip = log_RF2_masked[sclip_mask]

    idx  = np.digitize(log_RF2_masked_sclip,bins)
    running_median =[np.median(ratio_masked_sclip[idx==k]) for k in range(total_bins+1)[1:-1]]

    return frame, running_median, temp, np.median(log_RF2_masked_sclip)



# ------------------------------------------------------------------------------

def indiv_frame_plot(allframes, RF_list,RF_temps, ann_per,out_dir,total_bins,skip_idx):
    """Calculate dark rate ratio for every frame against reference frames

    For all dark frames, find the corresponding RF1 and RF2. Take the ratio of
    the frame over RF1 and take the log (base 10) of RF2. Mask out nan values
    and ratios below 0. Calculate a running median over `total_bins` separate
    bins across RF2.

    Parameters
    ----------
    allframes : dictionary
        dictionary of filenames which contain dark rate matrices
    RF_list : list
        List of Reference Frame tuples (RF1,RF2). Index corresponds to anneal
        period
    RF_temps : list
        List of Reference Frame temperature tuples (RF1,RF2). Index corresponds to anneal
        period
    ann_per : str
        string for ann label in form 'ann'+anneal period number e.g. 'ann1'
    out_dir : str
        string for output directory
    total_bins: int
        Total bins in log(darkrate) range
    skip_idx: list
        list of anneal periods skipped

    Returns
    -------
    None

    """
    ann_subset = [frame for frame in allframes.keys() if ann_per in frame]
    print(ann_subset,len(ann_subset))
    bins = np.linspace(-2.625,1.125, total_bins)
    delta = bins[1] - bins[0]
    for frame in ann_subset:
        if len([f for f in RF_list if f[0] == frame]) > 0 or len([f for f in RF_list if f[1] == frame]) > 0:
            continue
        f_name = frame.split('/')[-1]
        temp,num,ann,ext = f_name.split('_')
        ann_idx = int(ann[-1])-1
        for idx in skip_idx:
            if ann_idx >= idx:
                ann_idx-=1
        #dq_flags
        frame_flag = (np.ravel(fits.open(frame)['DQ'].data) !=16) & (np.ravel(fits.open(frame)['DQ'].data) != 256)
        RF1_flag = (np.ravel(fits.open(RF_list[ann_idx][0])['DQ'].data) !=16) & (np.ravel(fits.open(RF_list[ann_idx][0])['DQ'].data) != 256)
        dq_flag = np.array(frame_flag)*np.array(RF1_flag)
        frame_data = allframes.get(frame)[dq_flag]

        RF1 = allframes.get(RF_list[ann_idx][0])[dq_flag]
        RF2 = allframes.get(RF_list[ann_idx][1])[dq_flag]


        RF1_temp=RF_temps[ann_idx][0]
        RF2_temp=RF_temps[ann_idx][1]

        #Remove negative rates from RF2
        RF2_pos = RF2 > 0

        ratio = (frame_data / RF1)[RF2_pos]
        log_RF2 = np.log10(RF2[RF2_pos])

        #Remove Nan Values
        log_RF2_nonan = log_RF2[~np.isnan(log_RF2)]
        ratio_nonan = ratio[~np.isnan(log_RF2)]

        #Remove Negative Ratios
        ratio_mask = ratio_nonan >=0
        log_RF2_masked = log_RF2_nonan[ratio_mask]
        ratio_masked = ratio_nonan[ratio_mask]

        #Sigma Clip
        ratio_masked_sclip = sigma_clip(ratio_masked,sigma=3)
        ratio_masked_sclip = np.ma.compressed(ratio_masked_sclip)
        sclip_mask = np.isin(ratio_masked,ratio_masked_sclip)
        log_RF2_masked_sclip = log_RF2_masked[sclip_mask]

        idx  = np.digitize(log_RF2_masked_sclip,bins)
        print(idx)
        running_median = [np.median(ratio_masked_sclip[idx==k]) for k in range(total_bins+1)[1:-1]]

        import pdb;pdb.set_trace()

        fig, ax = plt.subplots(1,1,figsize = (15,10))
        ax.plot(log_RF2_masked_sclip,ratio_masked_sclip,'.',alpha=0.03,color='k')
        ax.plot(bins[:-1]+delta/2,running_median,'r--',lw=4,alpha=.8)
        ax.set_xlim(-3,1)
        ax.set_ylim(0,2)
        ax.set_xlabel(r'Log Dark Rate RF2 ({:{prec}} $\degree$C '.format(RF2_temp,prec='.6')+r'$s^{-1}$)',fontsize=24)
        ax.set_ylabel(r'{:{prec}} $\degree$C Frame : RF1 ({:{prec}} $\degree$C) Ratio'.format(temp,RF1_temp,prec='.6'),fontsize=24)
        ax.axhline(np.median(running_median), linewidth=2, color='k')

        plt.show()

# ------------------------------------------------------------------------------

def compare_RFs(allframes,RF_list,RF_temps):
    for ann,temps in zip(RF_list,RF_temps):
        import pdb;pdb.set_trace()
        fig, ax = plt.subplots(1,1,figsize=(15,10))
        RF1=allframes.get(ann[0])
        RF2=allframes.get(ann[1])
        ax.plot(RF1,RF2,'.',alpha=0.03, color='k')
        ax.set_xlabel(temps[0])
        ax.set_ylabel(temps[1])

        plt.show()

# ------------------------------------------------------------------------------

def slope_calc(median_matrix, frame_temp,total_bins,plot=False, store = False, out_name = 'output.csv'):
    """Calculate dark rate ratio for every frame against reference frames

    For all dark frames, find the corresponding RF1 and RF2. Take the ratio of
    the frame over RF1 and take the log (base 10) of RF2. Mask out nan values
    and ratios below 0. Calculate a running median over `total_bins` separate
    bins across RF2.

    Parameters
    ----------
    allframes : dictionary
        dictionary of filenames which contain dark rate matrices
    RF_list : list
        List of Reference Frame tuples (RF1,RF2). Index corresponds to anneal
        period.
    total_bins : int
        Total bins in log(darkrate) range
    plot : bool, optional
        Truth value for plotting results
    store : bool, optional
        Truth value for storing results
    out_name : str, optional
        Name of output csv file


    Returns
    -------
    median_matrix : list
        List of anneal period lists which contain running_median ratio lists
    frame_temp : list
        List of anneal period lists which contain temperatures for each frame
    total_bins : int
        Pass along the value of `total_bins`

    """
    print("--> Calculating with anneal shifts")
    #import pdb; pdb.set_trace()
    shift_temp = np.median([np.median([float(temp) for temp in ann]) for ann in frame_temp])
    print(shift_temp)

    indx_list = np.arange(0,total_bins-1,1)
    bins = np.linspace(-2.625,1.125, total_bins)
    slope_list = []
    rate_list=[]
    err_list = []
    if plot == True:
        fig = plt.figure(figsize=(15,10))
        count = 1

    for indx in indx_list:
        low_bound = bins[indx]
        up_bound = bins[indx+1]

        rate_list.append((up_bound+low_bound)/2)
        # Generate anneal periods with rate_ratio against temperature
        rate_ratio_list = [np.array([]) for _ in range(len(median_matrix))]
        temp_list = [np.array([]) for _ in range(len(frame_temp))]

        #populate anneal periods from median_matrix and frame_temp
        for i,ann in enumerate(median_matrix):
            for frame in ann:
                rate_ratio_list[i] = np.append(rate_ratio_list[i],float(frame[indx]))
        for i,ann in enumerate(frame_temp):
            for temp in ann:
                temp_list[i] = np.append(temp_list[i],float(temp))

        ann_fit_res=[]
        res_terms =[]
        ann_slopes=[]
        ann_n = []
        ann_ratio_shifted_arr = np.array([])
        ann_temp_arr = np.array([])
        n_total = 0 #Weighted mean slope combine

        if plot == True:
            ax = fig.add_subplot(3,5,count)
            ax.set_prop_cycle(cycler('color',['r','g','b','c','m','y']))
            ax.grid()
            count+=1
            ax.set_title('{} < log(rate) < {}'.format(low_bound,up_bound))
            ax.set_ylabel('Rate ratio')
            ax.set_xlabel('CCD Housing Temperature ('+r'$\degree$'+'C)')
        for ann_ratio, ann_temp in zip(rate_ratio_list, temp_list):

             #if len(ann_ratio) <= 4:
                 #continue

             regression = ols("data ~ x", data = dict(data=ann_ratio,x=ann_temp)).fit()
             test = regression.outlier_test()
             #import pdb; pdb.set_trace()
             outliers = ((ann_temp[i],ann_ratio[i]) for i,t in enumerate(test['bonf(p)'].tolist()) if t < 0.5)
             outcorr_temp = [ann_temp[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]
             outcorr_ratio = [ann_ratio[i] for i,t in enumerate(test['bonf(p)'].tolist()) if t > 0.5]
             if len(outcorr_ratio) <= 4:
                 continue
             if len(ann_ratio)-len(outcorr_ratio) != 0:
                 print("--> Rejected {} Outliers in log(rate)={} bin".format(len(ann_ratio)-len(outcorr_ratio),bins[indx]))

             temp_fit,res = np.polyfit(outcorr_temp,outcorr_ratio,1,cov=True)
             slope_err = np.sqrt(res[0][0])
             slope, intercept = np.poly1d(temp_fit)
             p = np.poly1d(temp_fit)


             shift = p(shift_temp)-1
             ann_ratio_shifted = np.array(outcorr_ratio)-shift
             ann_temp_elem = np.array(outcorr_temp)
             ann_ratio_shifted_arr = np.concatenate([ann_ratio_shifted_arr,ann_ratio_shifted])
             ann_temp_arr = np.concatenate([ann_temp_arr,ann_temp_elem])
             if plot == True:
                ax.plot(outcorr_temp,ann_ratio_shifted,'+',markersize=10)


        temp_fit,res = np.polyfit(ann_temp_arr,ann_ratio_shifted_arr,1,cov=True)
        slope_err = np.sqrt(res[0][0])
        slope, intercept = np.poly1d(temp_fit)
        p = np.poly1d(temp_fit)
        fitspace = np.linspace(16,24,30)
        linfit = p(fitspace)
        slope_list.append(slope)
        err_list.append(slope_err)

        if plot == True:
            #ax.plot(ann_temp_arr,ann_ratio_shifted_arr,'+',markersize=10)
            ax.plot(fitspace,linfit,'--',alpha=0.5,color="k")
            ax.text(0.1, 0.9, "Mean Slope: {:{prec}} +/- {:{prec}}".format(slope,slope_err,prec='.3'),
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,fontsize = 10)


    if plot == True:
        plt.tight_layout()
        plt.show()
        plt.savefig('RateRatio_vs_HousingTemp_PerRate_present.png')
    if store == True:
        rows  = zip(rate_list,slope_list,err_list)
        with open('results_redel/'+out_name,'w') as out_file:
            writer = csv.writer(out_file)
            for row in rows:
                writer.writerow(row)
            out_file.close()


        print("--> Writing output to "+out_name)
    return slope_list,rate_list,err_list

# ------------------------------------------------------------------------------

def plot_rate_dep(rate_list,slope_list,err_list):
    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(1,1,1)
    ax.grid()
    ax.errorbar(rate_list,slope_list,yerr=err_list,fmt='o-',color='k',alpha=0.7,capsize=7)
    ax.set_ylabel(r'Fractional change in dark rate per $\degree$C',fontsize=28)
    ax.set_xlabel(r'log rate ($e^-s^{-1}$)',fontsize=28)
    ax.set_ylim(0.03,0.10)
    ax.set_xlim(-3.0,1.0)

    plt.show()

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    ignore_folders = ["data/pipe_tweak","data/testingbiascorr"]
    store = True
    plot = False

    print("Loading yaml anneal boundaries...")
    with open('anneal_boundaries_v2.yml') as f:
        boundaries = yaml.load(f)

    if store:
        med_results = open("med_rates_redel.csv","w")

    for image_dir in glob.glob("data/*"):
        if image_dir in ignore_folders:
            continue
        image_dir = image_dir+'/'

    #for image_dir in ['data/sep02_dec02/']:
        dataset = "*flt.fits"

        out_dir = image_dir
        print("Retrieving Data from {}...".format(image_dir))
        image_subset,dateobs_subset, first_dark, last_dark = retrieve_data(dataset,
                                                                image_dir = image_dir)
        date_range = datetime.datetime.strftime(first_dark,'%Y_%m')+"_"+ datetime.datetime.strftime(last_dark,'%Y_%m')
        print("Data Retrieved! {} darks found.".format(len(image_subset)))

        print("Splitting into Anneal Periods...")
        ann_list = anneal_split(image_subset, dateobs_subset, date_start = first_dark,
                                date_end = last_dark, boundaries = boundaries)
        print("Successfully Split!")

        print("Calculating Housing Temperature...")
        df_ann_list = telem_temp(ann_list)
        print("Calculated Housing Temp for each Dark.")

        print("Clustering Darks for ocrrejection...")
        clust_ann_list = cluster_crj(df_ann_list)
        print("Successfully Clustered Darks.")

        print("Performing Cosmic Ray Rejection on Darks...")
        ann_labels = ocrreject_clusters(clust_ann_list, outdir = out_dir)
        print("Wrote crj files to "+out_dir)

        print("Selecting Reference Frames...")
        allframes, RF_list, RF_temps,skip_idx = choose_RF(ann_labels,out_dir)
        print("Reference Frames Selected")
        """
        print("Calculating Dark Rate Ratio for each frame...")
        median_matrix, frame_temp,total_bins,avg_darkrate = darkrate_ratio_calc(allframes,RF_list,skip_idx)
        print("Dark Rate Ratios calculated")
        """
        print("Calculating Dark Rate Ratio for each frame...")
        total_bins=16
        bins = np.linspace(-2.625,1.125, total_bins)
        delta = bins[1] - bins[0]

        p = Pool(8)
        results = p.map(frame_ratio_calc, allframes.keys())
        results = [row for row in results if row != None]
        p.close()

        RF_ann_labels = [RF[0].split('/')[-1].split('_')[2] for RF in RF_list]
        median_matrix = [
        [row[1] for row in results if (ann in row[0]) and (row != None)]
        for ann in RF_ann_labels]
        frame_temp = [
        [row[2] for row in results if (ann in row[0]) and (row != None)]
        for ann in RF_ann_labels]
        avg_darkrate = [
        [row[3] for row in results if (ann in row[0]) and (row != None)]
        for ann in RF_ann_labels]
        print("Dark Rate Ratios calculated")

        #compare_RFs(allframes,RF_list,RF_temps)
        #indiv_frame_plot(allframes,RF_list,RF_temps,'ann2',out_dir,total_bins,skip_idx)
        ann_darkrates=[]
        for idx,ann in enumerate(avg_darkrate):
            print("Anneal Period {} Average Log Dark Rate : {}".format(idx+1,np.mean(ann)))
            print("Fitting slopes to each anneal period in each dark rate region...")
            ann_darkrates.append(np.mean(ann))

        if store:
            med_results.write(date_range+","+str(np.mean(ann_darkrates))+'\n')

        out_name = date_range +".csv"
        slope_list,rate_list,err_list = slope_calc(median_matrix,frame_temp,total_bins,
                                                store = store,out_name = out_name,plot=plot)
        print("Mean Slope for each dark rate region calculated")

        if plot:
            print("Plotting...")
            plot_rate_dep(rate_list,slope_list,err_list)
