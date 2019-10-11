# --------------------------------------------------
# Evaluate BIOMARKEM patients local
#
# Sergi Valverde 2019
# svalverdee@eia.udg.edu
# --------------------------------------------------

from __future__ import print_function
import nibabel as nib
import os
import numpy as np
import sys

# --------------------------------------------------------------------------------
# EXPERIMENT SETTINGS
# - path to data
# - ...
# --------------------------------------------------------------------------------

IMAGE_PATH = '/mnt/DATA/w/biomarkem/images/skull/pacients_itinerants/all'
# IMAGE_PATH = '/mnt/DATA/w/biomarkem/images/skull/healthies/all'
scans = sorted(os.listdir(IMAGE_PATH))




# ------------------------------------------------------------------------------
# compute brain cavities for all subjects and scanners
#
# ------------------------------------------------------------------------------

bet_diff_cavity = []
robex_diff_cavity = []
parietal_diff_cavity = []
list_ias_bet_cavity = []
list_ias_robex_cavity = []
list_ias_parietal_cavity = []
list_tr_bet_cavity = []
list_tr_robex_cavity = []
list_tr_parietal_cavity = []
list_vh_bet_cavity = []
list_vh_robex_cavity = []
list_vh_parietal_cavity = []


for SCAN, index in zip(scans, range(len(scans))):

    # IAS
    current_scan = os.path.join(IMAGE_PATH, SCAN)
    bet_ias_scan = nib.load(os.path.join(current_scan, 'bet', 'ias_bet.nii.gz'))
    robex_ias_scan = nib.load(os.path.join(current_scan, 'robex', 'ias_robex.nii.gz'))
    parietal_ias_scan = nib.load(os.path.join(current_scan, 'parietal', 'ias_parietal.nii.gz'))
    bet_ias = bet_ias_scan.get_data()
    robex_ias = robex_ias_scan.get_data()
    parietal_ias = parietal_ias_scan.get_data()
    ias_vs = np.prod(bet_ias_scan.header['pixdim'][1:4])

    # TR
    current_scan = os.path.join(IMAGE_PATH, SCAN)
    bet_tr_scan = nib.load(os.path.join(current_scan, 'bet', 'tr_bet.nii.gz'))
    robex_tr_scan = nib.load(os.path.join(current_scan, 'robex', 'tr_robex.nii.gz'))
    parietal_tr_scan = nib.load(os.path.join(current_scan, 'parietal', 'tr_parietal.nii.gz'))
    bet_tr = bet_tr_scan.get_data()
    robex_tr = robex_tr_scan.get_data()
    parietal_tr = parietal_tr_scan.get_data()
    tr_vs = np.prod(bet_tr_scan.header['pixdim'][1:4])

    # VH
    current_scan = os.path.join(IMAGE_PATH, SCAN)
    bet_vh_scan = nib.load(os.path.join(current_scan, 'bet', 'vh_bet.nii.gz'))
    robex_vh_scan = nib.load(os.path.join(current_scan, 'robex', 'vh_robex.nii.gz'))
    parietal_vh_scan = nib.load(os.path.join(current_scan, 'parietal', 'vh_parietal.nii.gz'))
    bet_vh = bet_vh_scan.get_data()
    robex_vh = robex_vh_scan.get_data()
    parietal_vh = parietal_vh_scan.get_data()
    vh_vs = np.prod(bet_vh_scan.header['pixdim'][1:4])


    # compute brain cavities

    bet_ias_cavity = np.sum(bet_ias > 0) * ias_vs
    robex_ias_cavity = np.sum(robex_ias > 0) * ias_vs
    parietal_ias_cavity = np.sum(parietal_ias > 0) * ias_vs
    bet_tr_cavity = np.sum(bet_tr > 0) * tr_vs
    robex_tr_cavity = np.sum(robex_tr > 0) * tr_vs
    parietal_tr_cavity = np.sum(parietal_tr > 0) * tr_vs
    bet_vh_cavity = np.sum(bet_vh > 0) * vh_vs
    robex_vh_cavity = np.sum(robex_vh > 0) * vh_vs
    parietal_vh_cavity = np.sum(parietal_vh > 0) * vh_vs

    list_ias_bet_cavity.append(bet_ias_cavity)
    list_ias_robex_cavity.append(robex_ias_cavity)
    list_ias_parietal_cavity.append(parietal_ias_cavity)
    list_tr_bet_cavity.append(bet_tr_cavity)
    list_tr_robex_cavity.append(robex_tr_cavity)
    list_tr_parietal_cavity.append(parietal_tr_cavity)
    list_vh_bet_cavity.append(bet_vh_cavity)
    list_vh_robex_cavity.append(robex_vh_cavity)
    list_vh_parietal_cavity.append(parietal_vh_cavity)



# ------------------------------------------------------------------------------
# EXPERIMENT: differences (ml) in intracraneal cavities between scanners
#
# ------------------------------------------------------------------------------

print('\n\n--------------------------------------------------')
print('Differences in intracraneal cavity (ml)')
print('----------------------------------------------\n\n')


print('scan,', '|',
      'BET IAS/TR', '|',
      'BET IAS/VH', '|',
      'BET VH/TR', '|',
      'ACCUM BET', '|',
      'ROBEX IAS/TR', '|',
      'ROBEX IAS/VH', '|',
      'ROBEX VH/TR', '|',
      'ACCUM ROBEX', '|',
      'PARIETAL IAS/TR', '|',
      'PARIETAL IAS/VH', '|',
      'PARIETAL VH/TR', '|',
      'ACCUM PARIETAL', '|')

for SCAN, index in zip(scans, range(len(scans))):


    # differences between method (cavities)

    bet_ias_cavity = list_ias_bet_cavity[index]
    bet_tr_cavity = list_tr_bet_cavity[index]
    bet_vh_cavity = list_vh_bet_cavity[index]
    robex_ias_cavity = list_ias_robex_cavity[index]
    robex_tr_cavity = list_tr_robex_cavity[index]
    robex_vh_cavity = list_vh_robex_cavity[index]
    parietal_ias_cavity = list_ias_parietal_cavity[index]
    parietal_tr_cavity = list_tr_parietal_cavity[index]
    parietal_vh_cavity = list_vh_parietal_cavity[index]

    BET_IAS_TR = np.abs(bet_ias_cavity - bet_tr_cavity)
    BET_IAS_VH = np.abs(bet_ias_cavity - bet_vh_cavity)
    BET_VH_TR = np.abs(bet_vh_cavity - bet_tr_cavity)
    ACCUM_BET = BET_IAS_TR + BET_IAS_VH + BET_VH_TR

    ROBEX_IAS_TR = np.abs(robex_ias_cavity - robex_tr_cavity)
    ROBEX_IAS_VH = np.abs(robex_ias_cavity - robex_vh_cavity)
    ROBEX_VH_TR = np.abs(robex_vh_cavity - robex_tr_cavity)
    ACCUM_ROBEX = ROBEX_IAS_TR + ROBEX_IAS_VH + ROBEX_VH_TR

    PARIETAL_IAS_TR = np.abs(parietal_ias_cavity - parietal_tr_cavity)
    PARIETAL_IAS_VH = np.abs(parietal_ias_cavity - parietal_vh_cavity)
    PARIETAL_VH_TR = np.abs(parietal_vh_cavity - parietal_tr_cavity)
    ACCUM_PARIETAL = PARIETAL_IAS_TR + PARIETAL_IAS_VH + PARIETAL_VH_TR

    robex_diff_cavity.append(ACCUM_ROBEX / 1000)
    bet_diff_cavity.append(ACCUM_BET / 1000)
    parietal_diff_cavity.append(ACCUM_PARIETAL / 1000)

    print(SCAN, '|',
          np.round(BET_IAS_TR / 1000, 2), '|',
          np.round(BET_IAS_VH / 1000, 2), '|',
          np.round(BET_VH_TR / 1000, 2), '|',
          np.round(ACCUM_BET / 1000, 2),  '|',
          np.round(ROBEX_IAS_TR / 1000, 2), '|',
          np.round(ROBEX_IAS_VH / 1000, 2), '|',
          np.round(ROBEX_VH_TR / 1000, 2), '|',
          np.round(ACCUM_ROBEX / 1000, 2),  '|',
          np.round(PARIETAL_IAS_TR / 1000, 2), '|',
          np.round(PARIETAL_IAS_VH / 1000, 2), '|',
          np.round(PARIETAL_VH_TR / 1000, 2), '|',
          np.round(ACCUM_PARIETAL / 1000, 2),  '|')


# ------------------------------------------------------------------------------
# EXPERIMENT: percentage errors  in intracraneal cavities between scanners
#
# ------------------------------------------------------------------------------

print('\n\n--------------------------------------------------')
print('% error in intracraneal cavities')
print('----------------------------------------------\n\n')

print('scan,', '|',
      'BET IAS/TR', '|',
      'BET IAS/VH', '|',
      'BET VH/TR', '|',
      'ACCUM BET', '|',
      'ROBEX IAS/TR', '|',
      'ROBEX IAS/VH', '|',
      'ROBEX VH/TR', '|',
      'ACCUM ROBEX', '|',
      'PARIETAL IAS/TR', '|',
      'PARIETAL IAS/VH', '|',
      'PARIETAL VH/TR', '|',
      'ACCUM PARIETAL', '|')

for SCAN, index in zip(scans, range(len(scans))):



    # differences between method (cavities)

    bet_ias_cavity = list_ias_bet_cavity[index]
    bet_tr_cavity = list_tr_bet_cavity[index]
    bet_vh_cavity = list_vh_bet_cavity[index]
    robex_ias_cavity = list_ias_robex_cavity[index]
    robex_tr_cavity = list_tr_robex_cavity[index]
    robex_vh_cavity = list_vh_robex_cavity[index]
    parietal_ias_cavity = list_ias_parietal_cavity[index]
    parietal_tr_cavity = list_tr_parietal_cavity[index]
    parietal_vh_cavity = list_vh_parietal_cavity[index]


    # compute error

    ERROR_BET_IAS_TR = np.abs(bet_ias_cavity - bet_tr_cavity) / bet_tr_cavity * 100
    ERROR_BET_IAS_VH = np.abs(bet_ias_cavity - bet_vh_cavity) / bet_vh_cavity * 100
    ERROR_BET_VH_TR = np.abs(bet_vh_cavity - bet_tr_cavity) / bet_tr_cavity * 100
    ERROR_ACCUM_BET = ERROR_BET_IAS_TR + ERROR_BET_IAS_VH + ERROR_BET_VH_TR * 100
    ERROR_ROBEX_IAS_TR = np.abs(robex_ias_cavity - robex_tr_cavity) / robex_tr_cavity * 100
    ERROR_ROBEX_IAS_VH = np.abs(robex_ias_cavity - robex_vh_cavity) / robex_vh_cavity * 100
    ERROR_ROBEX_VH_TR = np.abs(robex_vh_cavity - robex_tr_cavity) / robex_tr_cavity * 100
    ERROR_ACCUM_ROBEX = ERROR_ROBEX_IAS_TR + ERROR_ROBEX_IAS_VH + ERROR_ROBEX_VH_TR * 100
    ERROR_PARIETAL_IAS_TR = np.abs(parietal_ias_cavity - parietal_tr_cavity) / parietal_tr_cavity * 100
    ERROR_PARIETAL_IAS_VH = np.abs(parietal_ias_cavity - parietal_vh_cavity) / parietal_vh_cavity * 100
    ERROR_PARIETAL_VH_TR = np.abs(parietal_vh_cavity - parietal_tr_cavity) / parietal_tr_cavity * 100
    ERROR_ACCUM_PARIETAL = ERROR_PARIETAL_IAS_TR + ERROR_PARIETAL_IAS_VH + ERROR_PARIETAL_VH_TR * 100

    print(SCAN, '|',
          np.round(ERROR_BET_IAS_TR / 1000, 5), '|',
          np.round(ERROR_BET_IAS_VH / 1000, 5), '|',
          np.round(ERROR_BET_VH_TR / 1000, 5), '|',
          np.round(ERROR_ACCUM_BET / 1000, 5),  '|',
          np.round(ERROR_ROBEX_IAS_TR / 1000, 5), '|',
          np.round(ERROR_ROBEX_IAS_VH / 1000, 5), '|',
          np.round(ERROR_ROBEX_VH_TR / 1000, 5), '|',
          np.round(ERROR_ACCUM_ROBEX / 1000, 5),  '|',
          np.round(ERROR_PARIETAL_IAS_TR / 1000, 5), '|',
          np.round(ERROR_PARIETAL_IAS_VH / 1000, 5), '|',
          np.round(ERROR_PARIETAL_VH_TR / 1000, 5), '|',
          np.round(ERROR_ACCUM_PARIETAL / 1000, 5),  '|')


# ------------------------------------------------------------------------------
# EXPERINMENT: scan_rescan analysis
# ------------------------------------------------------------------------------

print('\n\n--------------------------------------------------')
print('Intracraneal volume difference (ml) scan/rescan analysis')
print('----------------------------------------------\n\n')

print('scan', '|',
          'BET IAS', '|',
          'BET TR', '|',
          'BET VH', '|',
          'ACCUM BET', '|',
          'ROBEX IAS', '|',
          'ROBEX TR', '|',
          'ROBEX VH', '|',
          'ACCUM ROBEX', '|',
          'PARIETAL IAS', '|',
          'PARIETAL TR', '|',
          'PARIETAL VH', '|',
          'ACCUM PARIETAL', '|')

for index in range(0, len(scans), 2):
    b = index
    f = index +1
    SCAN = scans[b]

    d_bet_ias = np.abs(list_ias_bet_cavity[b] - list_ias_bet_cavity[f])
    d_robex_ias = np.abs(list_ias_robex_cavity[b] - list_ias_robex_cavity[f])
    d_parietal_ias = np.abs(list_ias_parietal_cavity[b] - list_ias_parietal_cavity[f])
    d_bet_tr = np.abs(list_tr_bet_cavity[b] - list_tr_bet_cavity[f])
    d_robex_tr = np.abs(list_tr_robex_cavity[b] - list_tr_robex_cavity[f])
    d_parietal_tr = np.abs(list_tr_parietal_cavity[b] - list_tr_parietal_cavity[f])
    d_bet_vh = np.abs(list_vh_bet_cavity[b] - list_vh_bet_cavity[f])
    d_robex_vh = np.abs(list_vh_robex_cavity[b] - list_vh_robex_cavity[f])
    d_parietal_vh = np.abs(list_vh_parietal_cavity[b] - list_vh_parietal_cavity[f])

    acc_bet = d_bet_ias + d_bet_tr + d_bet_vh
    acc_robex = d_robex_ias + d_robex_tr + d_robex_vh
    acc_parietal = d_parietal_ias + d_parietal_tr + d_parietal_vh

    print(SCAN, '|',
          np.round(d_bet_ias / 1000, 2), '|',
          np.round(d_bet_tr / 1000, 2), '|',
          np.round(d_bet_vh / 1000, 2), '|',
          np.round(acc_bet / 1000, 2), '|',
          np.round(d_robex_ias / 1000, 2), '|',
          np.round(d_robex_tr / 1000, 2), '|',
          np.round(d_robex_vh / 1000, 2), '|',
          np.round(acc_robex / 1000, 2), '|',
          np.round(d_parietal_ias / 1000, 2), '|',
          np.round(d_parietal_tr / 1000, 2), '|',
          np.round(d_parietal_vh / 1000, 2), '|',
          np.round(acc_parietal / 1000, 2), '|')

# ------------------------------------------------------------------------------
# EXPERINMENT: scan_rescan analysis (% error)
# ------------------------------------------------------------------------------

print('\n\n--------------------------------------------------')
print('Percentage error scan/rescan analysis')
print('----------------------------------------------\n\n')

print('scan', '|',
          'BET IAS', '|',
          'BET TR', '|',
          'BET VH', '|',
          'ACCUM BET', '|',
          'ROBEX IAS', '|',
          'ROBEX TR', '|',
          'ROBEX VH', '|',
          'ACCUM ROBEX', '|',
          'PARIETAL IAS', '|',
          'PARIETAL TR', '|',
          'PARIETAL VH', '|',
          'ACCUM PARIETAL', '|')


for index in range(0, len(scans), 2):
    b = index
    f = index +1
    SCAN = scans[b]

    d_bet_ias = np.abs(list_ias_bet_cavity[b] - list_ias_bet_cavity[f]) / list_ias_bet_cavity[f] * 100
    d_robex_ias = np.abs(list_ias_robex_cavity[b] - list_ias_robex_cavity[f])/ list_ias_robex_cavity[f] * 100
    d_parietal_ias = np.abs(list_ias_parietal_cavity[b] - list_ias_parietal_cavity[f])/ list_ias_parietal_cavity[f] * 100
    d_bet_tr = np.abs(list_tr_bet_cavity[b] - list_tr_bet_cavity[f])/ list_tr_bet_cavity[f] * 100
    d_robex_tr = np.abs(list_tr_robex_cavity[b] - list_tr_robex_cavity[f])/ list_tr_robex_cavity[f] * 100
    d_parietal_tr = np.abs(list_tr_parietal_cavity[b] - list_tr_parietal_cavity[f])/ list_tr_parietal_cavity[f] * 100
    d_bet_vh = np.abs(list_vh_bet_cavity[b] - list_vh_bet_cavity[f])/ list_vh_bet_cavity[f] * 100
    d_robex_vh = np.abs(list_vh_robex_cavity[b] - list_vh_robex_cavity[f])/ list_vh_robex_cavity[f] * 100
    d_parietal_vh = np.abs(list_vh_parietal_cavity[b] - list_vh_parietal_cavity[f])/ list_vh_parietal_cavity[f] * 100

    acc_bet = d_bet_ias + d_bet_tr + d_bet_vh
    acc_robex = d_robex_ias + d_robex_tr + d_robex_vh
    acc_parietal = d_parietal_ias + d_parietal_tr + d_parietal_vh

    print(SCAN, '|',
          np.round(d_bet_ias / 1000, 5), '|',
          np.round(d_bet_tr / 1000, 5), '|',
          np.round(d_bet_vh / 1000, 5), '|',
          np.round(acc_bet / 1000, 5), '|',
          np.round(d_robex_ias / 1000, 5), '|',
          np.round(d_robex_tr / 1000, 5), '|',
          np.round(d_robex_vh / 1000, 5), '|',
          np.round(acc_robex / 1000, 5), '|',
          np.round(d_parietal_ias / 1000, 5), '|',
          np.round(d_parietal_tr / 1000, 5), '|',
          np.round(d_parietal_vh / 1000, 5), '|',
          np.round(acc_parietal / 1000, 5), '|')