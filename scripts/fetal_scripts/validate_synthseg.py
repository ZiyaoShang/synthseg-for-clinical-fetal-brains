import os
from SynthSeg.validate import validate_training

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

validate_training(image_dir='/home/zshang/SP/data/grand_train_all/img_eval',
                    gt_dir=None,
                    models_dir='/home/zshang/SP/data/grand_train_all/SP_exp/model',
                    validation_main_dir='/home/zshang/SP/data/grand_train_all/SP_exp/validation',
                    labels_segmentation=[0,1,2,3,4,5,6,7],
                    n_neutral_labels=8,
                    evaluation_labels=[0,1,2,3,4,5,6,7],
                    step_eval=1,
                    min_pad=None,
                    cropping=None,
                    target_res=1.,
                    gradients=False,
                    flip=False,
                    topology_classes=None,
                    sigma_smoothing=0.5,
                    keep_biggest_component=True,
                    n_levels=5,
                    nb_conv_per_level=2,
                    conv_size=3,
                    unet_feat_count=24,
                    feat_multiplier=2,
                    activation='elu',
                    recompute=False)