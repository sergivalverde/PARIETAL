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
TEST_IMAGE_ROOT = '/mnt/DATA/w/NICMS2/images/CAMP/not'

options = {}

# training options: can come from different folders

options['training_path'] = os.path.join(TRAIN_IMAGE_ROOT, 'all')
options['input_data'] = ['T1.nii.gz', 'brainmask_template_SyN.nii.gz']
options['roi_mask'] = 'prebrainmask.nii.gz'
options['out_scan'] = 'brainmask_ss.nii.gz'

# experiment number
options['experiment'] = 'campinas_baseline_s2_multires_template_norm'
options['use_gpu'] = True

# computational resources
options['workers'] = 10
options['gpus'] = [2]

# other options fixed!
options['normalize'] = False
options['resample_epoch'] = False
options['data_augmentation'] = False
options['perc_training'] = 1
options['randomize_cases'] = True
options['input_channels'] = len(options['input_data'])
options['test_step'] = (16, 16, 16)
options['test_patch_shape'] = (32, 32, 32)

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

if __name__ == "__main__":
    """
    main function
    """
    parser = argparse.ArgumentParser(description='vox2vox implementation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='Select the GPU number to use (default=0)')
    opt = parser.parse_args()
    if opt.train:
        options['gpus'] = [opt.gpu]
        options['train'] = True
        train_skull_model(options)
    else:
        script_folder = os.path.dirname(os.path.abspath( __file__ ))
        options['model_path'] = os.path.join(script_folder, 'models')
        options['train'] = False
        run_skull_model(options)
