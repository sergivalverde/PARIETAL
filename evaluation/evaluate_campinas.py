# --------------------------------------------------
# Evaluate campinas dataset (GT images)
#
# Sergi Valverde 2020
# svalverdee@eia.udg.edu
# --------------------------------------------------

from __future__ import print_function
import nibabel as nib
import os
import numpy as np

import sys
sys.path.append('..')
from mri_utils.metrics import TPF_seg, FPF_seg, DSC_seg

# --------------------------------------------------
# parameters
IMAGE_PATH = '/home/sergivalverde/DATA/campinas/manual_test'
GTMASK = 'brainmask_gt.nii.gz'
# BRAINMASK = 'par_multires_s2_888.nii.gz'
BRAINMASK = 'par_multires_s2_int_8.nii.gz'
# BRAINMASK = 'baseline_s2.nii.gz'
PRINT_BY_SCAN = True
# ------------------------------------------------

# stats
dsc_list = []
sens_list = []
spec_list = []


scans = sorted(os.listdir(IMAGE_PATH))
for SCAN, index in zip(scans, range(len(scans))):

    gt_path = os.path.join(IMAGE_PATH, SCAN, GTMASK)
    brainmask_path = os.path.join(IMAGE_PATH, SCAN, BRAINMASK)

    # load scans
    seg = nib.load(brainmask_path).get_data() > 0.5
    gt = nib.load(gt_path).get_data() > 0

    # compute stats
    tpf = TPF_seg(gt, seg)
    fpf = FPF_seg(gt, seg)
    dsc = DSC_seg(gt, seg)

    dsc_list.append(dsc)
    sens_list.append(tpf)
    spec_list.append(1 - fpf)

    if PRINT_BY_SCAN:
        print(SCAN,
              "DSC", np.round(dsc, 2),
              "SENS", np.round(tpf, 2),
              "SPEC", np.round(1 - fpf, 2))

# print average values
print(".................................................")
print("|", np.round(np.mean(dsc_list), 5) * 100, '\pm', np.round(np.std(dsc_list), 5) * 100,
      "|", np.round(np.mean(sens_list), 5) * 100, '\pm', np.round(np.std(sens_list), 5) * 100,
      "|", np.round(np.mean(spec_list), 5) * 100, '\pm', np.round(np.std(spec_list), 5) * 100)
print(".................................................")
