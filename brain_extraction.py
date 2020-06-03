import os
import nibabel as nib
import numpy as np
from configparser import ConfigParser
from model import Parietal
from _utils.data_utils import reconstruct_image, extract_patches
from _utils.data_utils import get_voxel_coordenates
from _utils.processing import normalize_data
from scipy.ndimage import binary_fill_holes as fill_holes
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as lc


class BrainExtraction():
    """
    Documentation for BrainExtraction

    """
    def __init__(self,
                 model_name=None,
                 sampling_step=None,
                 patch_shape=None,
                 use_gpu=None,
                 gpu_number=None,
                 normalize=None,
                 threshold=None,
                 workers=None):

        # load otpions from config file
        self.path = os.path.dirname(os.path.abspath(__file__))

        opt = self.__load_options()

        self.t1_orig = None
        self.t1_orig_path = None
        self.t1_canonical = None
        self.t1_canonical_path = None
        self.scan_path = None
        self.scan_name = None
        self.output_path = None

        self.__normalize = opt['normalize'] if normalize is None else normalize
        self.__threshold = opt['out_threshold'] if threshold is None else threshold

        self.__input_channels = 1
        self.__scale = 2
        self.__step = opt['sampling_step'] if sampling_step is None else sampling_step
        self.__workers = opt['workers'] if workers is None else workers
        self.__model_name = opt['model_name'] if model_name is None else model_name
        self.__patch_shape = opt['patch_shape'] if patch_shape is None else patch_shape
        self.__use_gpu = opt['use_gpu'] if use_gpu is None else use_gpu
        self.__gpu_number = opt['gpu_number'] if gpu_number is None else [gpu_number]

        # initialize the model
        self.model = self.__initialize_model()

    def run(self, input_path, output_path, out_threshold=None):
        """
        Perform brain extraction to the input image

        inputs:
        - image_path = /path/to/the/input/image
        - ouput_path = /path/to/the/output/image
        - out_threshold = probability threshold to apply

        Outputs:
        - brainmask: np.array containing the brain roi.

        """
        scan_path, scan_name = os.path.split(input_path)
        self.scan_path = scan_path
        self.scan_name = scan_name
        self.output_path = output_path

        if out_threshold is not None:
            self.__threshold = out_threshold

        # pre-process image first
        self.__process_scan()

        # get input patches
        im_patches, ref_voxels = self.__get_patches()

        # predict brain cavity
        pred = self.model.test_net(im_patches)

        # reconstruction segmentation and save
        prob_skull = reconstruct_image(np.squeeze(pred),
                                       ref_voxels,
                                       self.t1_canonical.get_fdata().shape)

        # binarize the results and fill remaining holes
        brainmask = self.__post_process_skull(prob_skull > self.__threshold)

        # save the results
        brainmask = self.save_brainmask(brainmask)

        return brainmask

    def __process_scan(self):
        """
        to doc.
        Save the scan into a tmp folder
        """

        # check if tmp folder is available
        tmp_folder = os.path.join(self.scan_path, 'tmp')
        if os.path.exists(tmp_folder) is False:
            os.mkdir(tmp_folder)

        current_scan = os.path.join(self.scan_path, self.scan_name)
        nifti_orig = nib.load(current_scan)

        # assign the original t1 scan attribute to nifti_orig
        self.t1_orig = nifti_orig
        self.t1_orig_path = current_scan

        im_ = nifti_orig.get_fdata()

        processed_scan = nib.Nifti1Image(im_.astype('<f4'),
                                         affine=nifti_orig.affine)

        # check for extra dims
        if len(nifti_orig.get_fdata().shape) > 3:
            processed_scan = nib.Nifti1Image(
                np.squeeze(processed_scan.get_fdata()),
                affine=nifti_orig.affine)

        # normalize data between zero and one
        processed_scan.get_fdata()[:] = normalize_data(
            processed_scan.get_fdata(),
            norm_type='zero_one')

        t1_nifti_canonical = nib.as_closest_canonical(processed_scan)
        t1_nifti_canonical.to_filename(os.path.join(tmp_folder,
                                                    't1_can.nii.gz'))

        # assign the canoncal t1 attribute to t1_nifti_canonical
        self.t1_canonical = t1_nifti_canonical
        self.t1_canonical_path = os.path.join(tmp_folder, 't1_can.nii.gz')

    def __get_patches(self):
        """
        Extract data patches from the input image.

        - The head roi is computed as brain + skull (leaving air from the T1)
        - All the voxels inside the head roi are sampled according to
          the sampling options
        - Image patches are extracted

        A list of the image patches along with its center voxel coordinate is
        returned
        """

        # remove brainmask and obtain a whole head roi
        t1_scan = self.t1_canonical.get_fdata()
        head_roi = self.__compute_pre_mask(t1_scan)


        # get input_patches
        candidate_voxels = head_roi > 0
        ref_voxels = get_voxel_coordenates(t1_scan,
                                           candidate_voxels,
                                           step_size=self.__step)

        if self.__normalize:
            t1_scan = normalize_data(t1_scan)

        patches, _ = extract_patches(t1_scan,
                                     voxel_coords=ref_voxels,
                                     patch_size=self.__patch_shape,
                                     step_size=self.__step)

        patches = np.expand_dims(patches, axis=1)

        return patches, ref_voxels

    def save_brainmask(self, brainmask):
        """
        transmform brainmask from canonical to the original
        spacce before saving it to disk
        """

        brainmask_scan = nib.Nifti1Image(brainmask.astype('<f4'),
                                         affine=self.t1_canonical.affine)
        brainmask_nifti = self.__transform_canonical_to_orig(brainmask_scan,
                                                             self.t1_orig)
        brainmask_nifti.to_filename(self.output_path)

        return brainmask_nifti

    def __compute_pre_mask(self, input_mask, hist_bin=1):
        """
        Compute the ROI where brain intensities are (brain + skull).

        pre_mask = T1_input > min_intensity

        The minimum intensity is computed by taking the second bin in the
        histogram assuming the first one contains all the background parts

        input:
             T1_input: np.array containing the T1 image

        """
        hist, edges = np.histogram(input_mask, bins=64)
        pre_mask = input_mask > edges[hist_bin]

        return pre_mask

    def __load_model_conf(self):
        """
        to doc
        """

        CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
        user_config = ConfigParser.SafeConfigParser()
        user_config.read(os.path.join(CURRENT_PATH, 'config', 'config.cfg'))

        self.__normalize = user_config.get_boolean('data', 'normalize')
        self.__threshold = user_config.get('data', 'out_threshold')
        self.__workers = user_config.get('data', 'workers')
        self.__model_name = user_config.get('model', 'model_name')
        self.__scale = user_config.get('model', 'scale')
        self.__input_channels = user_config.get('model', 'input_channels')
        self.__step = (user_config.get('model', 'test_step'),
                       user_config.get('model', 'test_step'),
                       user_config.get('model', 'test_step'))
        self.__patch_shape = (user_config.get('model', 'patch_shape'),
                              user_config.get('model', 'patch_shape'),
                              user_config.get('model', 'patch_shape'))
        self.__use_gpu = user_config.get_boolean('model', 'use_gpu')
        self.__gpu_number = [user_config.get('model', 'gpu_number')]

    def __post_process_skull(self, input_mask):
        """
        post process input mask
        - fill holes in 2D
        - take the biggest region the final brainmask
        """

        # fill holes in 2D
        for s in range(input_mask.shape[2]):
            input_mask[:, :, s] = fill_holes(input_mask[:, :, s])

        # get the biggest region
        regions, num_regions = label(input_mask > 0)
        labels = np.arange(1, num_regions+1)
        output_mask = np.zeros_like(input_mask)
        max_region = np.argmax(
            lc(input_mask > 0, regions, labels, np.sum, int, 0)) + 1
        current_voxels = np.stack(np.where(regions == max_region), axis=1)
        output_mask[current_voxels[:, 0],
                    current_voxels[:, 1],
                    current_voxels[:, 2]] = 1

        return output_mask

    def __transform_canonical_to_orig(self, canonical, original):
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

    def __load_options(self):
        """
        Load  configuration file. Current configuration file is at
        config/config.cfg
        """
        options = {}

        user_config = ConfigParser()
        user_config.read(os.path.join(self.path, 'config', 'config.cfg'))

        # data options
        options['normalize'] = user_config.getboolean('data', 'normalize')
        options['out_threshold'] = user_config.getfloat('data', 'out_threshold')
        options['workers'] = user_config.getint('data', 'workers')

        # model options
        options['model_name'] = user_config.get('model', 'model_name')
        options['sampling_step'] = (user_config.getint('model', 'sampling_step'),
                                    user_config.getint('model', 'sampling_step'),
                                    user_config.getint('model', 'sampling_step'))
        options['patch_shape'] = (user_config.getint('model', 'patch_shape'),
                                  user_config.getint('model', 'patch_shape'),
                                  user_config.getint('model', 'patch_shape'))
        options['use_gpu'] = user_config.getboolean('model', 'use_gpu')
        options['gpu_number'] = [user_config.getint('model', 'gpu_number')]

        return options

    def __initialize_model(self):
        """
        Initialize the brain extraction model and load the weights
        stored in model_path/model_name
        """
        model_path = os.path.join(self.path, 'models')
        return Parietal(patch_shape=self.__patch_shape,
                        load_weights=True,
                        model_name=self.__model_name,
                        model_path=model_path,
                        scale=self.__scale,
                        gpu_mode=self.__use_gpu,
                        gpu_list=self.__gpu_number)
