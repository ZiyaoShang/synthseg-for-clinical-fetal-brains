"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# project imports
import tensorflow as tf
from SynthSeg.predict import predict
from scripts.fetal_scripts.ensemble import ensemble_combine, ensemble_combine_anyN
from scripts.fetal_scripts.fetal_helpers import resample_seg_according_to_base_vol, evaluate_own
import numpy as np
import os
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# tf.config.threading.set_inter_op_parallelism_threads(8)
# tf.config.threading.set_intra_op_parallelism_threads(8)
# inter_op_threads = tf.config.threading.get_inter_op_parallelism_threads()
# print(f"Inter-op parallelism threads: {inter_op_threads}")

# a list to all MRI datasets you would like to predict on:
path_images = ['/cluster/work/menze/zshang/data/CHUV/less_bg_with_extra_cereb/img',
'/cluster/work/menze/zshang/data/grand_train_all/processed_test/DHCP_PRETERM/img_T1']

# a list to all ground truth folders, set to None if you need to resample the segmentation before evaluating (i.e. if the direct network output is not in the same resolution as the input image). This happens if the corresponding entry in target_res is not None (see below).
gt_folder = [None, 
'/cluster/work/menze/zshang/data/grand_train_all/processed_test/DHCP_PRETERM/seg_T1']

# if the resolution of the testing set is very different from the training set, you would need to manually set the target_res to a value that is closer to the training resolution. In this case the testing MRI will be first resampled to the value you specified before being fed into the network. If the resolution of the testing set is similar to the training set, you can set target_res to None (which means no resampling to the input MRI will be done). 
target_res = [0.6, None]

# in the case where a manual target_res has been given for a dataset, specify its ground truth folder here. If corresponding target_res is None, set this to None as well. 
gt_folder_sampled_back = ["/cluster/work/menze/zshang/data/CHUV/less_bg_with_extra_cereb/seg", None]


start_from = 0 # we start from the first testing dataset
experiment_name1 = "noclst_origbg_noinflate_e14" # name for the first model for ensembling
experiment_name2 = "r_exp1_e15" # name for the second model for ensembling, basically a training repitition (please see paper for details) 
experiment_name3 = "r2_exp1_e18" # name for the third model for ensembling, basically a training repitition (please see paper for details)
dirs_exist = False # whether we need to create the directories for the output files
only_resamp_and_eval = False # if True, we only resample the outputs and evaluate the results. Set to true if you already have the network outputs. 
path_model1 = ['/cluster/work/menze/zshang/experiments/model/noclst_origbg_noinflate/dice_014.h5'] * len(path_images) # path to the first model for ensembling
path_model2 = ['/cluster/work/menze/zshang/experiments/model/r_exp1/dice_015.h5'] * len(path_images) # path to the second model for ensembling
path_model3 = ['/cluster/work/menze/zshang/experiments/model/r2_exp1/dice_018.h5'] * len(path_images) # path to the third model for ensembling


experiment_name = experiment_name1 + "_merge_" + experiment_name2 + "_merge_" + experiment_name3 # final experiments name would be the combination of the name for each model

# for each testing set, enter the base path where the results would be saved
path_segm_base = ['/cluster/work/menze/zshang/data/CHUV/experiments/results' + '/merged/' + experiment_name, 
'/cluster/work/menze/zshang/data/grand_train_all/processed_test/DHCP_PRETERM/res_T1' + '/merged/' + experiment_name]


path_post1 = [b + "/post1" for b in path_segm_base] # path to the posteriors of the first model (no need to change)
path_post2 = [b + "/post2" for b in path_segm_base] # path to the posteriors of the second model (no need to change)
path_post3 = [b + "/post3" for b in path_segm_base] # path to the posteriors of the third model (no need to change)
path_seg1 = [b + "/seg1" for b in path_segm_base] # path to the segmentations of the first model (no need to change)
path_seg2 = [b + "/seg2" for b in path_segm_base] # path to the segmentations of the second model (no need to change)
path_seg3 = [b + "/seg3" for b in path_segm_base] # path to the segmentations of the third model (no need to change)

# path to the max-posterior-merged segmentation: the path must be either--
# path_segm_base[index] + "/fin_seg" in the case where the input MRIs are NOT resampled before inference, or
# path_segm_base[index] + "/raw_seg" in the case where the input MRIs needs to be resampled before inference.
path_merged = ['/cluster/work/menze/zshang/data/CHUV/experiments/results' + '/merged/' + experiment_name + '/raw_seg', 
    '/cluster/work/menze/zshang/data/grand_train_all/processed_test/DHCP_PRETERM/res_T1' + '/merged/' + experiment_name + '/fin_seg']

# the segmentation labels used during training
path_segmentation_labels = np.array([0,1,2,3,4,5,6,7])


################################## please keep the following parameters unchanged ##################################

# a csv file that will contain the volumes of each segmented structure
path_vol = None
# a numpy array with the names corresponding to the structures in path_segmentation_labels
path_segmentation_names = None
# crop the input to a smaller shape for faster processing, or to make it fit on your GPU?
cropping = None
# if the input image is resampled, you have the option to save the resampled image.
path_resampled = None
# We can apply some test-time augmentation by flipping the input along the right-left axis and segmenting
# the resulting image. In this case, and if the network has right/left specific labels, it is also very important to
# provide the number of neutral labels. This must be the exact same as the one used during training.
flip = False
n_neutral_labels = None
# We can smooth the probability maps produced by the network. This doesn't change much the results, but helps to
# reduce high frequency noise in the obtained segmentations.
sigma_smoothing = 0.5
# Then we can operate some fancier version of biggest connected component, by regrouping structures within so-called
# "topological classes". For each class we successively: 1) sum all the posteriors corresponding to the labels of this
# class, 2) obtain a mask for this class by thresholding the summed posteriors by a low value (arbitrarily set to 0.1),
# 3) keep the biggest connected component, and 4) individually apply the obtained mask to the posteriors of all the
# labels for this class.
# Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
#                                             output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
#                                       topological_classes = [0,  0,  0,  1, 1, 2,  3,  1,  4,  4,  5,  6,  7]
# Here we regroup labels 2 and 3 in the same topological class, same for labels 41 and 42. The topological class of
# unsegmented structures must be set to 0 (like for 24 and 507).
topology_classes = None
# Finally, we can also operate a strict version of biggest connected component, to get rid of unwanted noisy label
# patch that can sometimes occur in the background. If so, we do recommend to use the smoothing option described above.
keep_biggest_component = True

# Regarding the architecture of the network, we must provide the predict function with the same parameters as during
# training.
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
activation = 'elu'
feat_multiplier = 2

# Also we can compute different surface distances (Hausdorff, Hausdorff99, Hausdorff95 and mean surface distance). The
# results will be saved in arrays similar to the Dice scores.
compute_distances = True

################################## please keep the above parameters unchanged ##################################

# make sure that the file are organized correctly 
assert len(path_images) == len(gt_folder)  
assert len(path_images) == len(path_model1) 
assert len(path_images) == len(path_model2) 
assert len(path_images) == len(path_model3) 
assert len(path_images) == len(target_res)
assert len(path_images) == len(path_post1)
assert len(path_images) == len(path_post2)
assert len(path_images) == len(path_post3)
assert len(path_images) == len(path_seg1)
assert len(path_images) == len(path_seg2)
assert len(path_images) == len(path_seg3)
assert len(path_images) == len(path_merged)
assert len(path_images) == len(gt_folder_sampled_back)

# assert np.all([("mri" in n) or ("img" in n) for n in path_images])
# assert np.all([(n == None) or ("seg" in n) for n in gt_folder])
# assert np.all([(n == None) or ("seg" in n) for n in gt_folder_sampled_back])
# assert np.all([("res" in n) for n in path_seg1])
# assert np.all([("res" in n) for n in path_seg2])
# assert np.all([("res" in n) for n in path_seg3])

assert experiment_name1.split('_')[-1][1:] in path_model1[0].split('/')[-1]
assert experiment_name2.split('_')[-1][1:] in path_model2[0].split('/')[-1]
assert experiment_name3.split('_')[-1][1:] in path_model3[0].split('/')[-1]
assert path_model1[0].split('/')[-2] in experiment_name1
assert path_model2[0].split('/')[-2] in experiment_name2
assert path_model3[0].split('/')[-2] in experiment_name3
assert len(np.unique(np.array(path_model1))) == 1
assert len(np.unique(np.array(path_model2))) == 1
assert len(np.unique(np.array(path_model3))) == 1

assert np.all([os.path.isdir(p) for p in path_images]) 
assert np.all([os.path.exists(p) for p in path_model1])
assert np.all([os.path.exists(p) for p in path_model2])
assert np.all([os.path.exists(p) for p in path_model3])

assert np.all([p.split("/")[-1] == "seg1" for p in path_seg1])
assert np.all([p.split("/")[-1] == "seg2" for p in path_seg2]) 
assert np.all([p.split("/")[-1] == "seg3" for p in path_seg3]) 
assert np.all([p.split("/")[-2] == experiment_name for p in path_seg1])
assert np.all([p.split("/")[-2] == experiment_name for p in path_seg2])
assert np.all([p.split("/")[-2] == experiment_name for p in path_seg3])
assert np.all([p.split("/")[-3] == "merged" for p in path_seg1])
assert np.all([p.split("/")[-3] == "merged" for p in path_seg2])
assert np.all([p.split("/")[-3] == "merged" for p in path_seg3])

assert np.all([p.split("/")[-1] == "post1" for p in path_post1])
assert np.all([p.split("/")[-1] == "post2" for p in path_post2]) 
assert np.all([p.split("/")[-1] == "post3" for p in path_post3]) 
assert np.all([p.split("/")[-2] == experiment_name for p in path_post1])
assert np.all([p.split("/")[-2] == experiment_name for p in path_post2])
assert np.all([p.split("/")[-2] == experiment_name for p in path_post3])
assert np.all([p.split("/")[-3] == "merged" for p in path_post1])
assert np.all([p.split("/")[-3] == "merged" for p in path_post2])
assert np.all([p.split("/")[-3] == "merged" for p in path_post3])

assert np.all([p.split("/")[-2] == experiment_name for p in path_merged])
assert np.all([p.split("/")[-3] == "merged" for p in path_merged])

for p in range(len(path_seg1)):
        if gt_folder[p] is not None:
                assert gt_folder_sampled_back[p] is None
                assert os.path.isdir(gt_folder[p])
                assert target_res[p] is None
                assert path_merged[p].split("/")[-1] == "fin_seg"
        else:
                assert gt_folder_sampled_back[p] is not None
                assert os.path.isdir(gt_folder_sampled_back[p])
                assert target_res[p] == 0.6 
                assert path_merged[p].split("/")[-1] == "raw_seg"              
        if not dirs_exist:
                assert not os.path.isdir(path_seg1[p])
                assert not os.path.isdir(path_seg2[p])
                assert not os.path.isdir(path_seg3[p])
                assert not os.path.isdir(path_post1[p])
                assert not os.path.isdir(path_post2[p])
                assert not os.path.isdir(path_post3[p])
                assert not os.path.isdir(path_merged[p])
        else:
                assert os.path.isdir(path_seg1[p])
                assert os.path.isdir(path_seg2[p])
                assert os.path.isdir(path_seg3[p])
                assert os.path.isdir(path_post1[p])
                assert os.path.isdir(path_post2[p])
                assert os.path.isdir(path_post3[p])
                assert os.path.isdir(path_merged[p])

# create the directories if needed
if not dirs_exist:
    print("dirs does not exist, creating dirs: ")
    for p in range(len(path_seg1)):
        os.makedirs(path_seg1[p])
        os.makedirs(path_seg2[p])
        os.makedirs(path_seg3[p])
        os.makedirs(path_post1[p])
        os.makedirs(path_post2[p])
        os.makedirs(path_post3[p])
        os.makedirs(path_merged[p])
else:
        print("dirs already exist: ")

print(path_seg1)
print(path_seg2)
print(path_seg3)
print(path_post1)
print(path_post2)
print(path_post3)
print(path_merged)

print("________________start predicting________________")

for index in range(start_from, len(gt_folder)):
        print(f"prediction #{str(index)}")
        print("path_images:")
        print(path_images[index])
        print("gt_folder:")
        print(gt_folder[index])
        print("target_res:")
        print(target_res[index])

        if not only_resamp_and_eval:
                print("prediction part 1: ")
                print("path_seg1:")
                print(path_seg1[index])
                print("path_model1:")
                print(path_model1[index])
                print("path_post1:")
                print(path_post1[index])

                predict(path_images[index],
                        path_seg1[index],
                        path_model1[index],
                        path_segmentation_labels,
                        n_neutral_labels=n_neutral_labels,
                        path_posteriors=path_post1[index],
                        path_resampled=path_resampled,
                        path_volumes=path_vol,
                        names_segmentation=path_segmentation_names,
                        cropping=cropping,
                        target_res=target_res[index],
                        flip=flip,
                        topology_classes=topology_classes,
                        sigma_smoothing=sigma_smoothing,
                        keep_biggest_component=keep_biggest_component,
                        n_levels=n_levels,
                        nb_conv_per_level=nb_conv_per_level,
                        conv_size=conv_size,
                        unet_feat_count=unet_feat_count,
                        feat_multiplier=feat_multiplier,
                        activation=activation,
                        gt_folder=gt_folder[index],
                        compute_distances=compute_distances)

                print("prediction part 2: ")
                print("path_seg2:")
                print(path_seg2[index])
                print("path_model2:")
                print(path_model2[index])
                print("path_post2:")
                print(path_post2[index])

                predict(path_images[index],
                        path_seg2[index],
                        path_model2[index],
                        path_segmentation_labels,
                        n_neutral_labels=n_neutral_labels,
                        path_posteriors=path_post2[index],
                        path_resampled=path_resampled,
                        path_volumes=path_vol,
                        names_segmentation=path_segmentation_names,
                        cropping=cropping,
                        target_res=target_res[index],
                        flip=flip,
                        topology_classes=topology_classes,
                        sigma_smoothing=sigma_smoothing,
                        keep_biggest_component=keep_biggest_component,
                        n_levels=n_levels,
                        nb_conv_per_level=nb_conv_per_level,
                        conv_size=conv_size,
                        unet_feat_count=unet_feat_count,
                        feat_multiplier=feat_multiplier,
                        activation=activation,
                        gt_folder=gt_folder[index],
                        compute_distances=compute_distances)

                print("prediction part 3: ")
                print("path_seg3:")
                print(path_seg3[index])
                print("path_model3:")
                print(path_model3[index])
                print("path_post3:")
                print(path_post3[index])

                predict(path_images[index],
                        path_seg3[index],
                        path_model3[index],
                        path_segmentation_labels,
                        n_neutral_labels=n_neutral_labels,
                        path_posteriors=path_post3[index],
                        path_resampled=path_resampled,
                        path_volumes=path_vol,
                        names_segmentation=path_segmentation_names,
                        cropping=cropping,
                        target_res=target_res[index],
                        flip=flip,
                        topology_classes=topology_classes,
                        sigma_smoothing=sigma_smoothing,
                        keep_biggest_component=keep_biggest_component,
                        n_levels=n_levels,
                        nb_conv_per_level=nb_conv_per_level,
                        conv_size=conv_size,
                        unet_feat_count=unet_feat_count,
                        feat_multiplier=feat_multiplier,
                        activation=activation,
                        gt_folder=gt_folder[index],
                        compute_distances=compute_distances)

                print("merging outputs: ")
                print("path_merged:")
                print(path_merged[index])
                print("posts used:")
                print([path_post1[index], path_post2[index], path_post3[index]])
                print("segs used:")
                print([path_seg1[index], path_seg2[index], path_seg3[index]])

                # ensemble_combine(path_post1=path_post1[index],
                #         path_post2=path_post2[index],
                #         path_seg1=path_seg1[index],
                #         path_seg2=path_seg2[index],
                #         path_merged=path_merged[index])

                ensemble_combine_anyN(
                    path_posts=[path_post1[index], path_post2[index], path_post3[index]],
                    path_segs=[path_seg1[index], path_seg2[index], path_seg3[index]],
                    path_merged=path_merged[index],
                    rule="max_posterior")

                print("deleting posterior files from " + path_post1[index] + " and " + path_post2[index] + " and " + path_post3[index])
                for file_path in glob.glob(os.path.join(path_post1[index], '*.nii.gz')):
                        if os.path.isfile(file_path):
                                os.remove(file_path) 
                        else:
                                print(file_path + " is not a .nii.gz file and thus not deleted")

                for file_path in glob.glob(os.path.join(path_post2[index], '*.nii.gz')):
                        if os.path.isfile(file_path):
                                os.remove(file_path) 
                        else:
                                print(file_path + " is not a .nii.gz file and thus not deleted")
                for file_path in glob.glob(os.path.join(path_post3[index], '*.nii.gz')):
                        if os.path.isfile(file_path):
                                os.remove(file_path) 
                        else:
                                print(file_path + " is not a .nii.gz file and thus not deleted")

        print("resampling if needed: ")
        if path_merged[index].split('/')[-1] == "raw_seg":
                resampled_save = os.path.join(path_merged[index][:-7],'fin_seg')
                print("need to resample")
                print("saved to " + resampled_save)
                resample_seg_according_to_base_vol(to_resample=[path_merged[index]], to_resample_file_extension='*_synthseg.nii.gz', base=path_images[index], base_file_extension='*_T2w.nii.gz', save_path=[resampled_save])
        else:
                assert path_merged[index].split('/')[-1] == "fin_seg"
                print("no need to resample")

        print("evaluate")
        if gt_folder[index] is not None:
                print("ground truth: " + gt_folder[index])
                print("evaluating: " + path_merged[index])
                evaluate_own(gt_dir=gt_folder[index], seg_dir=[path_merged[index]])
        else:
                resampled_save = os.path.join(path_merged[index][:-7], 'fin_seg')
                print("ground truth: " + gt_folder_sampled_back[index])
                print("evaluating: " + resampled_save)
                evaluate_own(gt_dir=gt_folder_sampled_back[index], seg_dir=[resampled_save])
        
        print("current dataset finished.")

print("done")
