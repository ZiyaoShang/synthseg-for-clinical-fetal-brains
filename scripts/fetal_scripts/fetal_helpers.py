
import nibabel as nib
import glob
import os
import numpy as np
from ext.lab2im.edit_volumes import align_volume_to_ref, crop_volume_around_region, pad_volume, resample_volume, resample_volume_like, crop_volume, crop_volume_with_idx
from ext.lab2im.utils import get_volume_info, save_volume, get_list_labels, list_images_in_folder
from SynthSeg.evaluate import evaluation
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def add_extra_cerebral_as_additional_label():
    print("add_extra_cerebral_as_additional_label()")
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_orig/seg" # path to the segmentation folder
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_orig/img" # path to the image folder
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_seg_exlabel" # path to save the new segmentations
    extra_label = 10 # label to set for the extra cerebral intensities

    seg_list = sorted(glob.glob(seg_path + '/*'))
    print(len(seg_list))
    img_list = sorted(glob.glob(img_path + '/*'))
    print(len(img_list))
    assert len(img_list) == len(seg_list)
    for i in range(len(img_list)):
        # print(seg_list[i].split('/')[-1].split('_')[0].split('-')[1])

        print("processing: " + str(img_list[i].split('/')[-1].split('_')[0]))
        # print(file_name)

        assert img_list[i].split('/')[-1].split('_')[0] == seg_list[i].split('/')[-1].split('_')[0]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],return_volume=True,aff_ref=None, max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],return_volume=True,aff_ref=None, max_channels=10)
        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.0001)), print(str(img_aff) + str(seg_aff))
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        # assert np.sum(img_im > 0) == np.sum(img_im != 0), np.min(img_im)
        if not (np.sum(img_im > 0) == np.sum(img_im != 0)):
            print("!!!reverting negative values: ")
            assert (np.all(img_im[img_im < 0] > -0.0001))
            img_im[img_im < 0] = 0.0
        assert np.all(seg_im >= 0)
        assert np.all(img_im >= 0)
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)
        assert isinstance(img_im_res[0], np.float32)
        assert isinstance(seg_im_res[0], np.float32)

        ext_ce = np.logical_and(seg_im == 0, img_im != 0)
        seg_im[ext_ce] = np.dtype(seg_im[0, 0, 0]).type(extra_label)

        file_name = os.path.join(save_path, seg_list[i].split('/')[-1])
        save_volume(volume=seg_im, aff=seg_aff, header=seg_h, path=file_name, res=seg_im_res, dtype="float64", n_dims=seg_n_dims)

def add_extra_cerebral_as_additional_label_synth():
    print("add_extra_cerebral_as_additional_label()")
    seg_path = "/Users/ziyaoshang/Desktop/zurich_synth/synth_1v1" # path to the segmentation folder
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/ZURICH/original_data/mri_train" # path to the image folder
    save_path = "/Users/ziyaoshang/Desktop/zurich_synth/synth_1v1_extracereb" # path to save the new segmentations
    extra_label = 10 # label to set for the extra cerebral intensities

    seg_list = sorted(glob.glob(seg_path + '/*'))
    img_list = sorted(glob.glob(img_path + '/*'))
    assert len(seg_list) / len(img_list) == 2
    for i in range(len(img_list)):
        for j in range(2):
            # print(seg_list[i].split('/')[-1].split('_')[0].split('-')[1])

            print("processing: " + str(seg_list[i*2+j].split('/')[-1]))
            print(img_list[i].split('/')[-1].split('_')[0])

            assert img_list[i].split('/')[-1].split('_')[0] == seg_list[i*2+j].split('/')[-1].split('_')[0]

            img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],return_volume=True,aff_ref=None, max_channels=10)
            seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i*2+j],return_volume=True,aff_ref=None, max_channels=10)
            assert (np.all(img_shp == seg_shp))
            assert (np.all(np.abs(img_aff - seg_aff) < 0.0001)), print(str(img_aff) + str(seg_aff))
            assert (img_n_dims == img_n_dims)
            assert (img_n_channels == seg_n_channels)
            assert (np.all(np.abs(img_im_res-seg_im_res) < 0.001)), str(img_im_res)+str(seg_im_res)
            assert np.sum(img_im > 0) == np.sum(img_im != 0)
            assert np.all(seg_im >= 0)
            assert np.all(img_im >= 0)
            assert isinstance(seg_im[0, 0, 0], np.float64)
            assert isinstance(img_im[0, 0, 0], np.float64)
            assert isinstance(img_im_res[0], np.float32)
            assert isinstance(seg_im_res[0], np.float32)

            ext_ce = np.logical_and(seg_im == 0, img_im != 0)
            seg_im[ext_ce] = np.dtype(seg_im[0, 0, 0]).type(extra_label)

            # print(np.sum(np.logical_and(seg_im != 0, img_im == 0)))
            # assert np.sum(img_im == 0) == np.sum(seg_im == 0), str(np.sum(img_im > 0)) + "  " + str(np.sum(seg_im > 0))
            # print(np.dtype(seg_im[0, 0, 0]).type(10))
            # print(np.average(img_im))
            file_name = os.path.join(save_path, seg_list[i*2+j].split('/')[-1])
            save_volume(volume=seg_im, aff=seg_aff, header=seg_h, path=file_name, res=seg_im_res, dtype="float64", n_dims=seg_n_dims)

def divide_bg_using_kmeans():
    print("divide_bg_using_kmeans()")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    all_bg_folder = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_seg_exlabel" # path to the segmentation folder already containing the extra cerebral label
    img_folder = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_orig/img" # path to the image folder
    save_path = '/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_seg_exlabel_bgsubd' # path to save the new segmentations
    prev_bg_label = 10 # (the only) label already set for the extra cerebral intensities
    bg_labels = [10, 11, 12, 13] # labels to set for all the extra cerebral intensities

    n_clusters = len(bg_labels)

    seg_list = sorted(glob.glob(all_bg_folder + '/*'))
    img_list = sorted(glob.glob(img_folder + '/*'))
    assert len(img_list) == len(seg_list)

    for i in range(len(img_list)):
        print("processing: " + str(img_list[i].split('/')[-1]))

        assert img_list[i].split('/')[-1].split('_')[1].split('-')[1] == \
               seg_list[i].split('/')[-1].split('_')[1].split('-')[1]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i],return_volume=True, aff_ref=None, max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],return_volume=True,aff_ref=None, max_channels=10)
        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.001)), img_aff - seg_aff
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        
        if not (np.sum(img_im > 0) == np.sum(img_im != 0)):
            print("!!!reverting negative values: ")
            assert (np.all(img_im[img_im < 0] > -0.0001))
            img_im[img_im < 0] = 0.0
        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)
        assert isinstance(img_im_res[0], np.float32)
        assert isinstance(seg_im_res[0], np.float32)

        # inds = np.where(np.logical_or(seg_im == 0, seg_im == 10))
        inds = np.where(seg_im == prev_bg_label)
        # print(img_im[inds])
        vects = np.column_stack([inds[0], inds[1], inds[2], img_im[inds]])
        for asse in range(vects.shape[0]):
            assert img_im[vects[asse,0].astype('int'), vects[asse,1].astype('int'), vects[asse,2].astype('int')] == vects[asse,3]
        normalized_data = MinMaxScaler().fit_transform(vects.copy()[:,-1].reshape(-1, 1))
        # normalized_data = MinMaxScaler().fit_transform(vects.copy())
        # normalized_data = vects.copy()
        # normalized_data[:, -1] *= 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++")
        kmeans.fit(normalized_data)
        labels = kmeans.labels_
        assert len(labels) == vects.shape[0]
        for l in range(labels.shape[0]):
            assert seg_im[tuple(vects[l, :3].astype('int'))] == prev_bg_label
            seg_im[tuple(vects[l, :3].astype('int'))] = bg_labels[labels[l]]

        # centers = kmeans.cluster_centers_
        file_name = os.path.join(save_path, seg_list[i].split('/')[-1])
        save_volume(volume=seg_im, aff=seg_aff, header=seg_h, path=file_name,
                    res=img_im_res, dtype="float64", n_dims=seg_n_dims)
        print("saved")
        

def center_labels():
    print("center_labels()")
    folder_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/dHCP_fetal_seg_exlabel_bgsubd" # path to the segmentation folder 
    file_extension = '*.nii.gz' # file extension to search for
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/trash2" # path to save the new segmentations
    allow_huge_input_sizes = True # whether to allow input sizes larger than 256x256x256


    files = sorted(glob.glob(os.path.join(folder_path, '**', file_extension), recursive=True))

    for file in files:
        # Load volumns
        im, shp, aff, n_dims, n_channels, h, im_res = get_volume_info(file, return_volume=True, aff_ref=None, max_channels=10)
        assert isinstance(im[0, 0, 0], np.float64)

        new_vol, cropping, new_aff = crop_volume_around_region(im,
                                                           mask=None,
                                                           masking_labels=None,
                                                           threshold=0.1,
                                                           margin=0,
                                                           cropping_shape=None,
                                                           cropping_shape_div_by=None,
                                                           aff=aff,
                                                           overflow='strict')
        # if any dimension of the original size of the labels is larger than 256, it would remain the original size, while the dimensions smaller than 256 would be padded to 256.
        final_vol, final_aff = pad_volume(new_vol, 256, padding_value=0, aff=new_aff, return_pad_idx=False)

        if not allow_huge_input_sizes:
            assert final_vol.shape == (256,256,256), final_vol.shape
        else: # Here, we center-crop the remaining large dimensions into 256
            if final_vol.shape != (256,256,256):
                print("large volumn: " + file)
                final_vol, final_aff = crop_volume(final_vol, cropping_margin=None, cropping_shape=(256,256,256), aff=final_aff, return_crop_idx=False, mode='center')
            else:
                print("normal volumn")
        
        assert final_vol.shape == (256,256,256), final_vol.shape
        save_file_name = os.path.join(save_path, file.split('/')[-1])

        save_volume(final_vol, final_aff, h, save_file_name, dtype='float64')


def remove_most_backgrounds_and_center_vol(
    seg_path = "/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/processed_others/CHUV/orig_seg", # path to the segmentation folder
    img_path = "/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/processed_others/CHUV/orig_img", # path to the image folder
    save_path = "/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/processed_others/CHUV/pros_img", # path to save the new images
    save_path2 = "/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/processed_others/CHUV/pros_seg", # path to save the new segmentations
    pad = False, # whether to pad the images to 160x160x160
    margin = 5 # margin to crop the images around the min bounding box
    ):

    seg_list = sorted(glob.glob(seg_path + '/*'))
    img_list = sorted(glob.glob(img_path + '/*'))

    assert len(img_list) == len(seg_list)
    splitby = "_rec-"

    for i in range(len(img_list)):
        if splitby is not None:
            print("processing: " + str(img_list[i].split('/')[-1].split(splitby)[0]))
            assert img_list[i].split('/')[-1].split(splitby)[0] == seg_list[i].split('/')[-1].split(splitby)[0], img_list[i].split('/')[-1].split(splitby)[0]

        img_im, img_shp, img_aff, img_n_dims, img_n_channels, img_h, img_im_res = get_volume_info(img_list[i], return_volume=True, aff_ref=None, max_channels=10)
        seg_im, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list[i],return_volume=True, aff_ref=None, max_channels=10)
        assert (np.all(img_shp == seg_shp))
        assert (np.all(np.abs(img_aff - seg_aff) < 0.001)), print(str(img_aff) + str(seg_aff))
        assert (img_n_dims == img_n_dims)
        assert (img_n_channels == seg_n_channels)
        assert (np.all(img_im_res == seg_im_res))
        if not (np.sum(img_im > 0) == np.sum(img_im != 0)):
            print("reverting negative values: ")
            assert (np.all(img_im[img_im < 0] > -0.001))
            img_im[img_im < 0] = 0.0

        assert isinstance(seg_im[0, 0, 0], np.float64)
        assert isinstance(img_im[0, 0, 0], np.float64)

        img_im, cropping, img_aff = crop_volume_around_region(img_im,
                                                               mask=None,
                                                               masking_labels=None,
                                                               threshold=0.1,
                                                               margin=margin,
                                                               cropping_shape=None,
                                                               cropping_shape_div_by=None,
                                                               aff=img_aff,
                                                               overflow='strict')

        # crop seg the using the same margin as the mri
        seg_im, seg_aff = crop_volume_with_idx(seg_im, cropping, aff=seg_aff, n_dims=seg_n_dims, return_copy=True)
        assert np.all(seg_im.shape == img_im.shape)

        if pad:
            final_img_im, final_img_aff = pad_volume(img_im, 160, padding_value=0, aff=img_aff, return_pad_idx=False)
            final_seg_im, final_seg_aff = pad_volume(seg_im, 160, padding_value=0, aff=seg_aff, return_pad_idx=False)
            assert np.all(final_seg_im.shape == final_img_im.shape)
            assert final_seg_im.shape == (160, 160, 160), final_seg_im.shape
        else:
            final_img_im, final_img_aff = img_im, img_aff
            final_seg_im, final_seg_aff = seg_im, seg_aff


        img_file_name = os.path.join(save_path, "lessbg" + img_list[i].split('/')[-1])
        seg_file_name = os.path.join(save_path2, "lessbg" + seg_list[i].split('/')[-1])

        # final_img_im, final_img_aff = img_im, img_aff
        # final_seg_im, final_seg_aff = seg_im, seg_aff

        save_volume(volume=final_img_im, aff=final_img_aff, header=img_h, path=img_file_name, res=img_im_res, dtype="float64", n_dims=img_n_dims)
        save_volume(volume=final_seg_im, aff=final_seg_aff, header=seg_h, path=seg_file_name, res=seg_im_res, dtype="float64", n_dims=seg_n_dims)


def resample_seg_according_to_base_vol(
    to_resample=['/cluster/work/menze/zshang/validation/synth_clstex1_origbg_noinflate_nonintense_ufc48/dice_0' + e for e in ['01', '02', '03', '04', '05', '06', '07','08','09','10','11','12','13','14','15','16','17','18','19','20']], # list of folders containing the segmentations to resample
    to_resample_file_extension = '*_synthseg.nii.gz', # file extension to search for the segmentations
    base='/cluster/work/menze/zshang/data/ZURICH/original_data/mri_train', # path to the folder containing the testing images
    base_file_extension = '*_T2w.nii.gz', # file extension to search for the testing images
    save_path=["/cluster/work/menze/zshang/validation/synth_clstex1_origbg_noinflate_nonintense_ufc48_finseg/dice_0" + e for e in ['01', '02', '03', '04', '05', '06', '07','08','09','10','11','12','13','14','15','16','17','18','19','20']] # path to save the resampled segmentations
    ):

    for ind in range(len(to_resample)):
        to_resample_files = sorted(glob.glob(os.path.join(to_resample[ind], '**', to_resample_file_extension), recursive=True))
        base_files = sorted(glob.glob(os.path.join(base, '**', base_file_extension), recursive=True))
        assert len(to_resample_files) == len(base_files)

        for i in range(len(to_resample_files)):
            im, shp, aff, n_dims, n_channels, h, im_res = get_volume_info(to_resample_files[i], return_volume=True, aff_ref=None, max_channels=10)
            b_im, b_shp, b_aff, b_n_dims, b_n_channels, b_h, b_im_res = get_volume_info(base_files[i], return_volume=True, aff_ref=None, max_channels=10)

            volume2 = resample_volume_like(vol_ref=b_im, aff_ref=b_aff, vol_flo=im, aff_flo=aff, interpolation='nearest')

            print(i)
            file_name = os.path.join(save_path[ind], to_resample_files[i].split('/')[-1])
            save_volume(volume=volume2, aff=b_aff, header=b_h, path=file_name, res=b_im_res, dtype="float64", n_dims=b_n_dims)


def evaluate_own(
        gt_dir='/cluster/work/menze/zshang/data/ZURICH/original_data/seg_train', # path to the ground truth folder
        seg_dir=["/cluster/work/menze/zshang/validation/synth_clstex1_origbg_noinflate_nonintense_ufc48_finseg/dice_0" + e for e in ['01', '02', '03', '04', '05', '06', '07','08','09','10','11','12','13','14','15','16','17','18','19','20']] # path to all segmentation folders
        ):
    for ind in range(len(seg_dir)):
        evaluation(gt_dir=gt_dir,
                seg_dir=seg_dir[ind],
                label_list=[0, 1, 2, 3, 4, 5, 6, 7],
                mask_dir=None,
                compute_score_whole_structure=False,
                path_dice=os.path.join(seg_dir[ind], 'dice.npy'),
                path_hausdorff=os.path.join(seg_dir[ind], 'hausdorff.npy'),
                path_hausdorff_99=os.path.join(seg_dir[ind], 'hausdorff_99.npy'),
                path_hausdorff_95=os.path.join(seg_dir[ind], 'hausdorff_95.npy'),
                path_mean_distance=os.path.join(seg_dir[ind], 'mean_distance.npy'),
                crop_margin_around_gt=10,
                list_incorrect_labels=None,
                list_correct_labels=None,
                use_nearest_label=False,
                recompute=True,
                verbose=True)


if __name__ == "__main__":

    ########### training label map manipulations ############

    # 1, create a new label representing extra cerebral intensities (where the label is 0 in the initial segmentation and the intensity is not 0 in the image)
    add_extra_cerebral_as_additional_label() # for a generic set of image/segmentation pairs
    add_extra_cerebral_as_additional_label_synth() # for our synthetic dataset

    # 2, subdivide the extra cerebral label into 4 labels using k-means clustering
    divide_bg_using_kmeans() 

    # 3, move the brains to the center of the volume 
    center_labels()

    ########### processing testing data ############
    # move the brains to the center of the volume and remove more backgrounds (absolute intensity < 0.1)
    remove_most_backgrounds_and_center_vol()

    ########### helper functions for inference and evaluation ############
    # 1, (if the testing image has been resampled before being fed into the model) resample the segmentation to the same affine space as the original testing image
    resample_seg_according_to_base_vol()
    # 2, evaluate the segmentation against the ground truth
    evaluate_own()

    pass