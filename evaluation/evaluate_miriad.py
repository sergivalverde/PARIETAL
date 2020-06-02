# --------------------------------------------------
# Evaluate MIRIAD subjects
#
# Sergi Valverde 2020
# svalverdee@eia.udg.edu
# --------------------------------------------------

import nibabel as nib
import os
import re
import numpy as np


def get_subject_information(scan_name):
    """
    Get relevant information from the subject ID.
    scan name is miriad_/pat_number/_/disease/_/sex/_/timepoint/_MR_/scan_num/

    input:

    """
    ind_ = [m.start() for m in re.finditer('_', scan_name)]

    scan_info = {}
    scan_info['scan_number'] = scan_name[ind_[0] + 1:ind_[1]]
    scan_info['disease_type'] = scan_name[ind_[1] + 1:ind_[2]]
    scan_info['subject_sex'] = scan_name[ind_[2] + 1:ind_[3]]
    scan_info['timepoint'] = int(scan_name[ind_[3] + 1:ind_[4]])
    scan_info['scan_num'] = int(scan_name[ind_[5] + 1:])

    return scan_info

# --------------------------------------------------------------------------------
# EXPERIMENT SETTINGS
# - path to data
# - ...
# --------------------------------------------------------------------------------


IMAGE_PATH = '/home/sergivalverde/DATA/miriad'
METHODS = ['BET', 'ROBEX', 'PARIETAL']
scans = sorted(os.listdir(IMAGE_PATH))

# brain_cavities = np.zeros((len(scans), len(METHODS), len(TIMEPOINTS)))

brain_cavities = np.zeros((len(METHODS), len(scans), 10, 10))

for s, SCAN in enumerate(scans):

    if s > 44:
        continue

    current_scan = os.path.join(IMAGE_PATH, SCAN)
    timepoints = sorted(os.listdir(current_scan))

    for m, method in enumerate(METHODS):

        # load each of the timepoints and compare it with respect to the rest
        # of the timepoints
        for t in timepoints:
            t_info = get_subject_information(t)
            t_num = t_info['timepoint']
            t_scan_num = t_info['scan_num']

            if t_scan_num > 1:
                continue

            # get the bbasal scan
            current_tm = os.path.join(current_scan, t)
            basal_image = nib.load(os.path.join(
                current_tm, method, 'brainmask.nii.gz')).get_fdata()

            for tt in timepoints:

                tt_info = get_subject_information(tt)
                tt_num = tt_info['timepoint']
                tt_scan_num = tt_info['scan_num']

                if tt_scan_num > 1:
                    continue

                # avoid processing the same case twice
                if tt_num <= t_num:
                    continue

                print("processing:", SCAN, '(', method, ')',  t, '-->',  tt)
                # get the followp up scan
                current_tm = os.path.join(current_scan, tt)
                followup_image = nib.load(os.path.join(
                    current_tm, method, 'brainmask.nii.gz')).get_fdata()

                # compute diffs
                current_dif = abs(np.sum(basal_image) - np.sum(followup_image))
                brain_cavities[m, s, t_num - 1, tt_num - 1] = current_dif
                brain_cavities[m, s, tt_num - 1, t_num - 1] = current_dif
