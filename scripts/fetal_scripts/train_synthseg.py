"""

This script shows how we trained SynthSeg.
Importantly, it reuses numerous parameters seen in the previous tutorial about image generation
(i.e., 2-generation_explained.py), which we strongly recommend reading before this one.



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

import os
import sys
import numpy as np
import tensorflow as tf

# project imports
from SynthSeg.training import training
# from scripts.fetal_scripts.feature_extractor import extract

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

inter_op_threads = tf.config.threading.get_inter_op_parallelism_threads()
print(f"Inter-op parallelism threads: {inter_op_threads}")

# path training label maps
path_training_label_maps = "/Users/ziyaoshang/Desktop/fa2023/SP/SP/data/ZURICH/original_data/seg"
path_model_dir = "/Users/ziyaoshang/Desktop/fa2023/SP/all_segs/trash"
# path_model_dir = '/home/zshang/SP/data/ZURICH/experiments/model/delete'
batchsize = 1

# architecture parameters
n_levels = 5           # number of resolution levels
nb_conv_per_level = 2  # number of convolution per level
conv_size = 3          # size of the convolution kernel (e.g. 3x3x3)
unet_feat_count = 24   # number of feature maps after the first convolution
activation = 'elu'     # activation for all convolution layers except the last, which will use softmax regardless
feat_multiplier = 2    # if feat_multiplier is set to 1, we will keep the number of feature maps constant throughout the network; 2 will double them(resp. half) after each max-pooling (resp. upsampling); 3 will triple them, etc.

# training parameters
lr = 1e-4               # learning rate
wl2_epochs = 1          # number of pre-training epochs with wl2 metric w.r.t. the layer before the softmax
# IMPORTANT: set this to zero if using checkpoint !!!!!
dice_epochs = 20        # number of training epochs
steps_per_epoch = 5000  # number of iteration per epoch
checkpoint = None
# checkpoint = '/home/zshang/SP/data/ZURICH/experiments/model/delete/dice_001.h5'
# checkpoint = None

# ---------- Generation parameters ----------
# these parameters are from the previous tutorial, and thus we do not explain them again here

# generation and segmentation labels
path_generation_labels = np.array([0,10,11,12,13,1,2,3,4,5,6,7])
n_neutral_labels = 12
path_segmentation_labels = np.array([0,0,0,0,0,1,2,3,4,5,6,7])

# shape and resolution of the outputs
target_res = None
output_shape = 160
# important!!! output_div_by_n == 2 ** n_levels
n_channels = 1

# GMM sampling
prior_distributions = 'uniform'
path_generation_classes = None

# spatial deformation parameters
flipping = False
scaling_bounds = 0.2
rotation_bounds = 20
shearing_bounds = .012
translation_bounds = False
nonlin_std = 4.
bias_field_std = .7

# acquisition resolution parameters
randomise_res = True

# note the background is not set to black

inflate = False
# labels = layers.InflateLabels(labels_to_inflate=[1,2,3,4,5,6,7], inflate_val=[1,1,1,2,3,4], name='labels_inflate')(labels)
weighted_sampling = False

# ------------------------------------------------------ weighted sampling  ------------------------------------------------------
subjects_prob = None
if weighted_sampling:
    subjects_prob = extract(seg_path=path_training_label_maps, labels_all=path_generation_labels, inner_labels=[2,3,4,5,6,7], n_clusters=3)

# ------------------------------------------------------ Training ------------------------------------------------------

print("__commence trianing__")
training(path_training_label_maps,
         path_model_dir,
         generation_labels=path_generation_labels,
         segmentation_labels=path_segmentation_labels,
         subjects_prob=subjects_prob,
         n_neutral_labels=n_neutral_labels,
         batchsize=batchsize,
         n_channels=n_channels,
         target_res=target_res,
         output_shape=output_shape,
         prior_distributions=prior_distributions,
         generation_classes=path_generation_classes,
         flipping=flipping,
         scaling_bounds=scaling_bounds,
         rotation_bounds=rotation_bounds,
         shearing_bounds=shearing_bounds,
         translation_bounds=translation_bounds,
         nonlin_std=nonlin_std,
         randomise_res=randomise_res,
         bias_field_std=bias_field_std,
         n_levels=n_levels,
         nb_conv_per_level=nb_conv_per_level,
         conv_size=conv_size,
         unet_feat_count=unet_feat_count,
         feat_multiplier=feat_multiplier,
         activation=activation,
         lr=lr,
         wl2_epochs=wl2_epochs,
         dice_epochs=dice_epochs,
         steps_per_epoch=steps_per_epoch,
         checkpoint=checkpoint,
         inflate=inflate)
