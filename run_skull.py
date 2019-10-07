# --------------------------------------------------
# PARIETAL: DeeP leArning mRI brain ExTrAction Tool
#
# Yet another brain extration tool for MRI
#
# Sergi Valverde 2019
# --------------------------------------------------

import os
import argparse
from base import run_skull_model
# --------------------------------------------------
# OPTIONS
# --------------------------------------------------


# training options

if __name__ == "__main__":

    """
    main function
    """

    # Training settings
    parser = argparse.ArgumentParser(
        description= "PARIETAL: yet another deeP leARnIng brain ExTrAtion tooL")
    parser.add_argument('--input_image',
                        action='store',
                        help='T1 nifti image to process (mandatory)')
    parser.add_argument('--out_name',
                        action='store',
                        help='Output name for the resulted skull stripped image (mandatory)')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='Output threshold to binarize the brainmask (default=0.5)')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPU for computing (default=false)')
    parser.add_argument('--gpu_number',
                        type=int,
                        default=0,
                        help='Select the GPU number to use (default=0)')
    parser.add_argument('--model_path',
                        default=None,
                        help='Absolute path to the trained model to use (default=campinas baseline)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbose mode')
    options = {}
    opt = parser.parse_args()

    # check input file from opt.input_image, whether it's a input_file
    # or relative / absolute path
    script_folder = os.path.dirname(os.path.abspath( __file__ ))
    input_image = opt.input_image

    if str.find(input_image, '/') >= 0:
        if os.path.isabs(input_image):
            (im_path, im_name) = os.path.split(input_image)
            options['test_path'] = im_path
            options['input_data'] = [im_name]
        else:
            (im_path, im_name) = os.path.split(input_image)
            options['test_path'] = os.path.join(os.getcwd(), im_path)
            options['input_data'] = [im_name]
    else:
        options['test_path'] = os.getcwd()
        options['input_data'] = [opt.input_image]

    # model options
    if opt.model_path is not None:
        if str.find(opt.model_path, '/') >= 0:
            (options['model_path'],
             options['experiment']) = os.path.split(opt.model_path)
        else:
            options['model_path'] = os.path.join(script_folder, 'models')
            options['experiment'] = opt.model_path
    else:
        options['model_path'] = os.path.join(script_folder, 'models')
        options['experiment'] = 'campinas_baseline'

    print(options['model_path'], options['experiment'])
    options['out_name'] = opt.out_name
    options['out_threshold'] = opt.threshold
    options['use_gpu'] = opt.gpu
    options['gpus'] = [opt.gpu_number]
    options['input_channels'] = len(options['input_data'])
    options['workers'] = 10
    options['verbose'] = opt.verbose    # test the skull model

    run_skull_model(options)
