import os
import shutil
import time
import nibabel as nib
import numpy as np
import random
import ants
from torch.utils.data import DataLoader
from mri_utils.data_utils import reconstruct_image, extract_patches
from mri_utils.data_utils import get_voxel_coordenates
from mri_utils.processing import normalize_data  # n4_normalization
from model import SkullNet
from scipy.ndimage import binary_fill_holes as fill_holes
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as lc
from dataset import MRI_DataPatchLoader
from dataset import RotatePatch, FlipPatch
from pyfiglet import Figlet


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

    for scan in list_scans:
        image_path = os.path.join(options['training_path'], scan)
        if os.path.exists(os.path.join(image_path, 'tmp')) is False:
            os.mkdir(os.path.join(image_path, 'tmp'))

        current_scan = os.path.join(image_path, options['input_data'][0])
        T1 = nib.load(current_scan)
        T1.get_data()[:] = compute_pre_mask(T1.get_data())
        T1.to_filename(os.path.join(image_path, 'tmp', options['roi_mask']))

    # move training scans to tmp a folder before building the MRI_PatchLoader
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
    transform = []
    if options['data_augmentation']:
        transform += [RotatePatch(90),
                      RotatePatch(180),
                      FlipPatch(0),
                      FlipPatch(180)]

    # dataset
    training_dataset = MRI_DataPatchLoader(input_data,
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
    skull_net = SkullNet(input_channels=options['input_channels'],
                         patch_shape=options['train_patch_shape'],
                         scale=options['scale'],
                         model_name=options['experiment'],
                         gpu_mode=options['use_gpu'],
                         gpu_list=options['gpus'])

    skull_net.load_weights()
    skull_net.train_model(t_dataloader, v_dataloader)


def run_skull_model(options):
    """
    helper function that loads the UNET model and the model weights.
    It calls the =test_image= function afterwards.
    """

    # skull net

    options['normalize'] = True
    options['test_step'] = (16, 16, 16)
    options['scale'] = 2
    options['train_patch_shape'] = (32, 32, 32)
    options['test_patch_shape'] = (32, 32, 32)
    options['test_step'] = (16, 16, 16)

    # model_path = os.path.dirname(os.path.abspath( __file__ ))

    skull_net = SkullNet(input_channels=options['input_channels'],
                         patch_shape=options['train_patch_shape'],
                         model_name=options['experiment'],
                         model_path=options['model_path'],
                         scale=options['scale'],
                         gpu_mode=options['use_gpu'],
                         gpu_list=options['gpus'])

    if options['verbose']:
        show_info(options)

    print("Initializing model....", end=' ')
    skull_net.load_weights()
    print("done")

    # perform inference
    # infer_image_reg(skull_net, options)
    infer_image(skull_net, options)
    # test_on_batch(skull_net, options)


def infer_image(net, options):
    """
    Perform inference using data from testing folder on a single image
    Steps for each testing image:
    - Load bathches
    - Perform inference
    - Reconstruct the output image
    """
    patch_shape = options['test_patch_shape']
    step = options['test_step']
    scan_path = options['test_path']

    scan_time = time.time()

    # 1. transform images to canonical space
    # --------------------------------------------------
    transform_input_images(scan_path, options['input_data'])

    # 2. get candidate voxels
    # --------------------------------------------------
    t1_tmp_path = os.path.join(scan_path,
                               'tmp',
                               options['input_data'][0])

    t1_nifti_canonical = nib.load(t1_tmp_path)
    mask_image = compute_pre_mask(t1_nifti_canonical.get_data())
    ref_mask, ref_voxels = get_candidate_voxels(mask_image,
                                                step,
                                                sel_method='all')

    test_patches = get_data_channels(os.path.join(scan_path, 'tmp'),
                                     options['input_data'],
                                     ref_voxels,
                                     patch_shape,
                                     step,
                                     normalize=options['normalize'])

    # 3. predict skull
    # --------------------------------------------------
    print("Predicting skull...", end=' '),
    pred = net.test_net(test_patches)

    # 4. reconstruction segmentation and save
    # --------------------------------------------------
    prob_skull = reconstruct_image(np.squeeze(pred),
                                   ref_voxels,
                                   ref_mask.shape)

    # binarize the results and fill remaining holes
    brainmask = prob_skull > options['out_threshold']

    # brainmask = fill_holes(brainmask)
    brainmask = post_process_skull(brainmask)

    brainmask_can = nib.Nifti1Image(brainmask.astype('<f4'),
                                    affine=t1_nifti_canonical.affine)
    brainmask_can.to_filename(os.path.join(scan_path,
                                           'tmp',
                                           'brainmask_can.nii.gz'))

    T1_scan = nib.load(os.path.join(scan_path, options['input_data'][0]))
    brainmask_nifti = transform_canonical_to_orig(brainmask_can,
                                                  T1_scan)

    brainmask_nifti.to_filename(os.path.join(scan_path,
                                             options['out_name'] + '.nii.gz'))

    shutil.rmtree(os.path.join(scan_path, 'tmp'))
    print("done")
    print('Elapsed time', np.round(time.time() - scan_time, 2), 'sec')


def infer_image_reg(net, options):
    """
    Perform inference using data from testing folder on a single image
    Steps for each testing image:
    - Load bathches
    - Perform inference
    - Reconstruct the output image
    """
    patch_shape = options['test_patch_shape']
    step = options['test_step']
    scan_path = options['test_path']

    scan_time = time.time()

    # load input images and transform it to the canonical space
    # all images are stored in a tmp folder
    # --------------------------------------------------
    # 1. transform the images
    # 2. register masks
    # 3. process masks (skull)
    # 4. invert registration
    # --------------------------------------------------

    # 1. register mask and store as
    orig_scan = os.path.join(scan_path, options['input_data'][0])
    T1_scan = nib.load(orig_scan)
    my_tx = register_native_to_template(orig_scan)
    my_tx['warpedmovout'].to_filename(
        os.path.join(scan_path, 'tmp', 't1_reg.nii.gz'))

    # 2. transform to cannonical
    t1_reg = nib.load(os.path.join(scan_path, 'tmp', 't1_reg.nii.gz'))
    t1_nifti_canonical = nib.as_closest_canonical(t1_reg)
    t1_nifti_canonical.to_filename(os.path.join(
        scan_path,
        'tmp',
        't1_reg_can.nii.gz'))

    # 3. get candidate voxels
    # input images stacked as channels
    mask_image = compute_pre_mask(t1_nifti_canonical.get_data())
    ref_mask, ref_voxels = get_candidate_voxels(mask_image,
                                                step,
                                                sel_method='all')
    test_patches = get_data_channels(os.path.join(scan_path, 'tmp'),
                                     ['t1_reg_can.nii.gz'],

                                     ref_voxels,
                                     patch_shape,
                                     step,
                                     normalize=options['normalize'])

    # 4. predict skull
    print("Predicting skull...", end=' '),
    pred = net.test_net(test_patches)

    # reconstruction segmentation
    prob_skull = reconstruct_image(np.squeeze(pred),
                                   ref_voxels,
                                   ref_mask.shape)

    # binarize the results and fill remaining holes
    brainmask = prob_skull > options['out_threshold']
    brainmask = fill_holes(brainmask)
    # T1_skulled = T1_reg_image * brainmask

    bm_reg_can_path = os.path.join(scan_path,
                                   'tmp',
                                   'brainmask_reg_can.nii.gz')
    brainmask_can = nib.Nifti1Image(brainmask.astype('<f4'),
                                    affine=t1_nifti_canonical.affine)
    brainmask_can.to_filename(bm_reg_can_path)

    brainmask_nifti = transform_canonical_to_orig(brainmask_can,
                                                  t1_reg)
    bm_reg_path = os.path.join(scan_path, 'tmp', 'brainmask_reg.nii.gz')
    brainmask_nifti.to_filename(bm_reg_path)

    im = transform_template_to_native(orig_scan,
                                      os.path.join(scan_path,
                                                   'tmp',
                                                   't1_reg.nii.gz'),
                                      bm_reg_can_path)

    T1_out = nib.Nifti1Image(im, affine=T1_scan.affine)
    T1_out.to_filename(os.path.join(scan_path,
                                    options['out_name'] + '.nii.gz'))

    shutil.rmtree(os.path.join(scan_path, 'tmp'))
    print("done")
    print('Elapsed time', np.round(time.time() - scan_time, 2), 'sec')


def get_data_channels(image_path,
                      scan_names,
                      ref_voxels,
                      patch_shape,
                      step,
                      normalize=False):
    """
    Get data for each of the channels
    """
    out_patches = []
    for s in scan_names:
        current_scan = os.path.join(image_path, s)
        patches, _ = get_input_patches(current_scan,
                                       ref_voxels,
                                       patch_shape,
                                       step,
                                       normalize=normalize)
        out_patches.append(patches)

    return np.concatenate(out_patches, axis=1)


def get_input_patches(scan_path,
                      ref_voxels,
                      patch_shape,
                      step,
                      normalize=False,
                      expand_dims=True):
    """
    get current patches for a given scan
    """
    # current_scan = nib.as_closest_canonical(nib.load(scan_path)).get_data()
    current_scan = nib.load(scan_path).get_data()

    if normalize:
        current_scan = normalize_data(current_scan)

    patches, ref_voxels = extract_patches(current_scan,
                                          voxel_coords=ref_voxels,
                                          patch_size=patch_shape,
                                          step_size=step)

    if expand_dims:
        patches = np.expand_dims(patches, axis=1)

    return patches, ref_voxels


def get_candidate_voxels(input_mask,  step_size, sel_method='all'):
    """
    Extract candidate patches.
    """

    if sel_method == 'all':
        candidate_voxels = input_mask > 0

        voxel_coords = get_voxel_coordenates(input_mask,
                                             candidate_voxels,
                                             step_size=step_size)
    return candidate_voxels, voxel_coords


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

    # Nyul normalization. we need a brainmask to improve normalization
    # The CAMP351 brainmask registered against the T1 is used as a brain mask
    # current_path = get_current_path()
    # normalization_hist = current_path + '/normalization/campinas_histogram.npy'
    # brainmask = register_brainmask_template_to_native(os.path.join(
    #     image_path, scan_names[0]))

    # normalize images
    for s in scan_names:
        current_scan = os.path.join(image_path, s)
        nifti_orig = nib.load(current_scan)

        # check for extra dims
        if len(nifti_orig.get_data().shape) > 3:
            nifti_orig = nib.Nifti1Image(np.squeeze(nifti_orig.get_data()),
                                         affine=nifti_orig.affine)

        nifti_orig.get_data()[:] = normalize_data(nifti_orig.get_data(),
                                                  norm_type='zero_one')

        # nifti_orig.get_data()[:] = nyul_apply_standard_scale(nifti_orig.get_data(),
        # normalization_hist,

        t1_nifti_canonical = nib.as_closest_canonical(nifti_orig)
        t1_nifti_canonical.to_filename(os.path.join(tmp_folder, s))
        # nifti_orig.to_filename(os.path.join(tmp_folder, s))


def register_brainmask_template_to_native(scan_path):
    """
    Transform input images for processing. Processed
    images are stored in the tmp folder
    - register CC351_t1_template to the native space. Using 'SyN' by default
    - reshape the CC351_brainmask to the native space
    """

    current_script = get_current_path()
    im_template = ants.image_read(os.path.join(current_script,
                                               'campinas_template',
                                               'CC351_nonlin_t1.nii.gz'))
    brain_template = ants.image_read(
        os.path.join(current_script,
                     'campinas_template',
                     'CC351_nonlin_brainmask.nii.gz'))

    # check if tmp folder is available
    (image_path, scan_name) = os.path.split(scan_path)

    tmp_folder = os.path.join(image_path, 'tmp')
    if os.path.exists(tmp_folder) is False:
        os.mkdir(tmp_folder)

    # register the template against the original image
    im_orig = ants.image_read(scan_path)
    my_tx = ants.registration(im_orig,
                              im_template,
                              type_of_transform='SyN')

    # reg_iterations=(160, 80, 40))
    #
    warped_template = my_tx['warpedmovout']

    # apply the obtained transformation to the template_brainmask
    warped_brainmask = ants.apply_transforms(im_orig,
                                             brain_template,
                                             my_tx['fwdtransforms'])

    # store the results
    warped_brainmask.to_filename(os.path.join(image_path,
                                              'tmp',
                                              'template_brainmask.nii.gz'))
    # store the results
    warped_template.to_filename(os.path.join(image_path,
                                             'tmp',
                                             'template_t1.nii.gz'))

    return warped_brainmask.numpy()


def register_native_to_template(scan_path):
    """
    native to template
    """

    current_script = get_current_path()
    im_template = ants.image_read(os.path.join(current_script,
                                               'campinas_template',
                                               'CC351_nonlin_t1.nii.gz'))
    # check if tmp folder is available
    (image_path, scan_name) = os.path.split(scan_path)

    tmp_folder = os.path.join(image_path, 'tmp')
    if os.path.exists(tmp_folder) is False:
        os.mkdir(tmp_folder)

    # register the template against the original image
    im_orig = ants.image_read(scan_path)
    my_tx = ants.registration(im_template,
                              im_orig,
                              type_of_transform='Affine')

    return my_tx


def transform_template_to_native(original_image_path,
                                 moved_image_path,
                                 brainmask_path):

    """
    transform template to native back
    """

    im_orig = ants.image_read(original_image_path)
    im_moved = ants.image_read(moved_image_path)
    bm_moved = ants.image_read(brainmask_path)
    # register the template against the original image

    my_tx = ants.registration(im_orig,
                              im_moved,
                              type_of_transform='Affine')

    warped_image = ants.apply_transforms(im_orig,
                                         bm_moved,
                                         my_tx['invtransforms'])

    return warped_image.numpy()


def get_current_path():
    """
    Just get the path to where this script is
    """
    return os.path.dirname(os.path.abspath(__file__))


def show_info(options):
    """
    Show method information
    """
    f = Figlet(font="slant")
    print("--------------------------------------------------")
    print(f.renderText("PARIETAL"))
    print("Yet another deeP leARnIng brain ExTrAtion tooL")
    print("(c) Sergi Valverde, 2020")
    print(" ")
    print("version: v0.2")
    print("--------------------------------------------------")
    print(" ")
    print("Image information:")
    print("input path: ", options['test_path'])
    print("input image: ", options['input_data'][0])
    print("Output image: ", options['out_name'])
    print("GPU using:", options['use_gpu'])
    print(" ")
    print("Model information")
    print("Model path:", options['model_path'])
    print("Model name:", options['experiment'])
    print("--------------------------------------------------")


def post_process_skull(input_mask):
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
