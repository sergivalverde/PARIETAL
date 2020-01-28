# --------------------------------------------------
# Skull stripping experiments

#
# Sergi Valverde 2019
#
# --------------------------------------------------

import os
import argparse
from base import train_skull_model, run_skull_model

# --------------------------------------------------
# set experimental parameters
# --------------------------------------------------
TRAIN_IMAGE_ROOT = '/home/sergivalverde/DATA/campinas'
# TEST_IMAGE_ROOT = '/mnt/DATA/w/NICMS2/images/BETCNN_Sergi'
# TEST_IMAGE_ROOT = '/mnt/DATA/w/NICMS2/images/WMH'
TEST_IMAGE_ROOT = '/mnt/DATA/w/NICMS2/images/CAMP/manual_test'
# TEST_IMAGE_ROOT = '/mnt/DATA/w/NICMS2/images/OASIS/'
# TEST_IMAGE_ROOT = '/mnt/DATA/w/SKULL/images/LPBA40/native_space_radio'
# TEST_IMAGE_ROOT = '/mnt/DATA/w/SKULL/images/OASIS/disc1'
# --------------------------------------------------

options = {}
# experiment number

options['input_data'] = ['T1_hm.nii.gz']
options['out_name'] = 'T1_skull_cnn'
options['out_threshold'] = 0.5
options['test_path'] = os.path.join(TEST_IMAGE_ROOT)
options['experiment'] = 'campinas_baseline_hm'
options['use_gpu'] = True

# computational resources

options['workers'] = 10
options['gpus'] = [2]

# other options fixed!
options['normalize'] = True
options['resample_epoch'] = False
options['data_augmentation'] = False
options['perc_training'] = 1
options['randomize_cases'] = True
options['input_channels'] = len(options['input_data'])
options['test_step'] = (16, 16, 16)
options['scale'] = 1
options['test_patch_shape'] = (32, 32, 32)

# training options
options['training_path'] = os.path.join(TRAIN_IMAGE_ROOT, 'all')
# options['roi_mask'] = 'prebrainmask.nii.gz' # TODO reparar el training
options['out_scan'] = 'brainmask_ss.nii.gz'
options['train_patch_shape'] = (32, 32, 32)
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

if __name__ == "__main__":
    """
    main function
    """
    parser = argparse.ArgumentParser(description='vox2vox implementation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    opt = parser.parse_args()
    if opt.train:
        options['train'] = True
        train_skull_model(options)
    else:
        options['train'] = False
        run_skull_model(options)
