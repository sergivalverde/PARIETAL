# --------------------------------------------------
# Skull stripping experiments
# Train a model
#
# Sergi Valverde 2020
#
# --------------------------------------------------

import os
import argparse
import nibabel as nib
import numpy as np
import random
from model import Parietal
from torch.utils.data import DataLoader
from _utils.processing import normalize_data
from _utils.dataset import MRI_DataPatchLoader


def train_skull_model(options):
    """
    Train skull-stripping model

    Input images are transformed into the cannnical space before training.
    """

    # list all training scans in a list
    list_scans = sorted(os.listdir(options['training_path']))

    if options['randomize_cases']:
        random.shuffle(list_scans)

    list_scans = list_scans[:int(len(list_scans) * options['perc_training'])]
    t_delimiter = int(len(list_scans) * (1 - options['train_split']))
    training_data = list_scans[:t_delimiter]
    validation_data = list_scans[t_delimiter:]

    # precompute brain masks
    print('--------------------------------------------------')
    print('PREPROCESSING DATA')
    print('--------------------------------------------------')

    # compute the prebrainmask to guide patch extraction
    options['roi_mask'] = 'prebrainmask.nii.gz'

    if options['preprocess']:
        for scan in list_scans:
            image_path = os.path.join(options['training_path'], scan)
            if os.path.exists(os.path.join(image_path, 'tmp')) is False:
                os.mkdir(os.path.join(image_path, 'tmp'))
            current_scan = os.path.join(image_path, options['input_data'][0])
            T1 = nib.load(current_scan)
            T1.get_data()[:] = compute_pre_mask(T1.get_data())
            T1.to_filename(os.path.join(image_path,
                                        'tmp',
                                        options['roi_mask']))

        # move training scans to tmp a folder before building the PatchLoader
        for scan in list_scans:
            scan_names = options['input_data'] + [options['out_scan']] + [options['roi_mask']]
            current_scan = os.path.join(options['training_path'], scan)
            transform_input_images(current_scan, scan_names)

    print('--------------------------------------------------')
    print('TRAINING DATA:')
    print('--------------------------------------------------')

    input_data = {scan: [os.path.join(options['training_path'], scan, 'tmp', d)
                         for d in options['input_data']]
                  for scan in training_data}

    labels = {scan: [os.path.join(options['training_path'],
                                  scan,
                                  'tmp',
                                  options['out_scan'])]
              for scan in training_data}

    rois = {scan: [os.path.join(options['training_path'],
                                scan,
                                'tmp',
                                options['roi_mask'])]
            for scan in training_data}

    # data augmentation
    set_transforms = None

    # dataset
    training_dataset = MRI_DataPatchLoader(
        input_data,
        labels,
        rois,
        patch_size=options['train_patch_shape'],
        sampling_step=options['training_step'],
        sampling_type=options['sampling_type'],
        normalize=options['normalize'],
        transform=set_transforms)

    t_dataloader = DataLoader(training_dataset,
                              batch_size=options['batch_size'],
                              shuffle=True,
                              num_workers=options['workers'])

    print('--------------------------------------------------')
    print('VALIDATION DATA:')
    print('--------------------------------------------------')

    input_data = {scan: [os.path.join(options['training_path'], scan, 'tmp', d)
                         for d in options['input_data']]
                  for scan in validation_data}
    labels = {scan: [os.path.join(options['training_path'],
                                  scan,
                                  'tmp',
                                  options['out_scan'])]
              for scan in validation_data}
    rois = {scan: [os.path.join(options['training_path'],
                                scan,
                                'tmp',
                                options['roi_mask'])]
            for scan in validation_data}

    validation_dataset = MRI_DataPatchLoader(
        input_data,
        labels,
        rois,
        patch_size=options['train_patch_shape'],
        sampling_step=options['training_step'],
        sampling_type=options['sampling_type'],
        normalize=options['normalize'],
        transform=set_transforms)

    v_dataloader = DataLoader(validation_dataset,
                              batch_size=options['batch_size'],
                              shuffle=True,
                              num_workers=options['workers'])

    # train the model
    p = Parietal(input_channels=options['input_channels'],
                 patch_shape=(32, 32, 32),
                 scale=options['scale'],
                 model_name=options['experiment'],
                 gpu_mode=options['use_gpu'],
                 gpu_list=options['gpus'])

    p.train_model(t_dataloader, v_dataloader)


def transform_canonical_to_orig(canonical, original):
    """
    Transform back a nifti file that has been moved to the canonical space

    This function is a bit hacky, but so far it's the best way to deal with
    transformations between datasets without registration
    """

    ori2can = nib.io_orientation(original.affine)

    # transform the canonical image back to the original space
    ori2ori = nib.io_orientation(canonical.affine)
    can2ori = nib.orientations.ornt_transform(ori2ori, ori2can)
    return canonical.as_reoriented(can2ori)


def compute_pre_mask(T1_input, hist_bin=1):
    """
    Compute the ROI where brain intensities are (brain + skull).

    pre_mask = T1_input > min_intensity

    The minimum intensity is computed by taking the second bin in the histogram
    assuming the first one contains all the background parts

    input:
       T1_input: np.array containing the T1 image
       bin_edge: histogram bin number
    """

    hist, edges = np.histogram(T1_input, bins=64)
    pre_mask = T1_input > edges[hist_bin]

    return pre_mask


def transform_input_images(image_path, scan_names):
    """
    Transform input input images for processing
    + n4 normalization between scans
    + move t1 to the  canonical space
    Images are stored in the tmp/ folder
    """

    # check if tmp folder is available
    tmp_folder = os.path.join(image_path, 'tmp')
    if os.path.exists(tmp_folder) is False:
        os.mkdir(tmp_folder)

    # normalize images
    for s in scan_names:
        current_scan = os.path.join(image_path, s)
        nifti_orig = nib.load(current_scan)
        im_ = nifti_orig.get_data()

        processed_scan = nib.Nifti1Image(im_.astype('<f4'),
                                         affine=nifti_orig.affine)

        # check for extra dims
        if len(nifti_orig.get_data().shape) > 3:
            processed_scan = nib.Nifti1Image(
                np.squeeze(processed_scan.get_data()),
                affine=nifti_orig.affine)

        processed_scan.get_data()[:] = normalize_data(
            processed_scan.get_data(),
            norm_type='zero_one')

        t1_nifti_canonical = nib.as_closest_canonical(processed_scan)
        t1_nifti_canonical.to_filename(os.path.join(tmp_folder, s))


def get_current_path():
    """
    Just get the path to where this script is
    """
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    """
    main function
    """

    # --------------------------------------------------
    # set experimental parameters
    # --------------------------------------------------
    TRAIN_IMAGE_ROOT = '/home/sergivalverde/DATA/campinas'
    options = {}

    # training options: can come from different folders
    options['training_path'] = os.path.join(TRAIN_IMAGE_ROOT, 'all')
    options['input_data'] = ['T1.nii.gz']
    options['out_scan'] = 'brainmask_ss.nii.gz'

    # experiment number
    options['experiment'] = 'campinas_baseline_s2_multires'
    options['use_gpu'] = True

    # computational resources
    options['workers'] = 10
    options['gpus'] = [2]

    # other options fixed!
    options['preprocess'] = False
    options['normalize'] = False
    options['resample_epoch'] = False
    options['data_augmentation'] = False
    options['perc_training'] = 1
    options['randomize_cases'] = True
    options['input_channels'] = len(options['input_data'])
    options['train_patch_shape'] = (32, 32, 32)
    options['scale'] = 2
    options['training_step'] = (16, 16, 16)
    options['patch_threshold'] = 0.1
    options['num_epochs'] = 200
    options['batch_size'] = 32
    options['train_split'] = 0.2
    options['patience'] = 50
    options['l_weight'] = 10
    options['resume_training'] = False
    options['sampling_type'] = 'balanced+roi'
    options['min_sampling_th'] = 0.1
    options['verbose'] = 1

    parser = argparse.ArgumentParser(
        description='PARIETAL: yet Another Skull Stripping Method')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='Select the GPU number to use (default=0)')

    opt = parser.parse_args()
    options['gpus'] = [opt.gpu]
    train_skull_model(options)
