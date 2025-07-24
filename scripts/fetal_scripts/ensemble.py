import numpy as np
import os
import glob
from ext.lab2im.utils import get_volume_info, save_volume

def ensemble_combine(
    path_post1, # path to the first set of posteriors
    path_post2, # path to the second set of posteriors
    path_seg1, # path to the first set of segmentations
    path_seg2, # path to the second set of segmentations
    path_merged): # path to save the merged segmentations
    """ 
    Combine two sets of segmentations using the maximum-posterior rule.
    """

    post1 = sorted(glob.glob(path_post1 + '/*.nii.gz'))
    post2 = sorted(glob.glob(path_post2 + '/*.nii.gz'))
    seg1 = sorted(glob.glob(path_seg1 + '/*.nii.gz'))
    seg2 = sorted(glob.glob(path_seg2 + '/*.nii.gz'))

    assert len(post1) == len(post2)
    assert len(post1) == len(seg1)
    assert len(post1) == len(seg2)
    
    for ind in range(len(seg1)):

        print("processing: " + str(seg1[ind].split('/')[-1]))

        seg1_vol, seg1_shp, seg1_aff, seg1_n_dims, seg1_n_channels, seg1_h, seg1_im_res = get_volume_info(seg1[ind], return_volume=True, aff_ref=None, max_channels=10)
        seg2_vol, seg2_shp, seg2_aff, seg2_n_dims, seg2_n_channels, seg2_h, seg2_im_res = get_volume_info(seg2[ind], return_volume=True, aff_ref=None, max_channels=10)

        post1_vol, post1_shp, post1_aff, post1_n_dims, post1_n_channels, post1_h, post1_im_res = get_volume_info(post1[ind], return_volume=True, aff_ref=None, max_channels=10)
        post2_vol, post2_shp, post2_aff, post2_n_dims, post2_n_channels, post2_h, post2_im_res = get_volume_info(post2[ind], return_volume=True, aff_ref=None, max_channels=10)

        assert seg1_shp == seg2_shp
        assert (np.all(np.abs(seg1_aff - seg2_aff) < 0.0001))
        assert seg1_n_dims == seg2_n_dims
        assert seg1_n_channels == seg2_n_channels
        assert (np.all(seg1_im_res == seg2_im_res))
        assert isinstance(seg1_vol[0, 0, 0], np.float64)

        post1_vol = np.max(post1_vol, axis=-1, out=None, keepdims=False)
        post2_vol = np.max(post2_vol, axis=-1, out=None, keepdims=False)

        new_vol = np.where(post1_vol>post2_vol, seg1_vol, seg2_vol)

        save_path = os.path.join(path_merged, "merged_" + seg1[ind].split('/')[-1])
        save_volume(volume=new_vol, aff=seg1_aff, header=seg1_h, path=save_path, res=seg1_im_res, dtype='float64', n_dims=seg1_n_dims)


def ensemble_combine_anyN(
    path_posts, # list of paths to the sets of posteriors
    path_segs, # list of paths to the sets of segmentations
    path_merged, # path to save the merged segmentations
    rule="max_posterior"): # rule to combine the segmentations
    """ 
    Combine any number of sets of segmentations using the maximum-posterior rule.
    """

    assert len(path_posts) == len(path_segs)
    all_posts = [] # shape: (#models, #templates)
    all_segs = [] # shape: (#models, #templates)

    for i in range(len(path_posts)):
        assert len(glob.glob(path_posts[i] + '/*.nii.gz')) == len(glob.glob(path_segs[i] + '/*.nii.gz'))
        all_posts.append(sorted(glob.glob(path_posts[i] + '/*.nii.gz')))
        all_segs.append(sorted(glob.glob(path_segs[i] + '/*.nii.gz')))

    all_posts = np.array(all_posts)
    all_segs = np.array(all_segs)
    for templ_ind in range(all_posts.shape[1]): # for each template
        path_segs_templ = all_segs[:, templ_ind]
        path_posts_templ = all_posts[:, templ_ind] # path to segmentations for this particular template from all models
        assert len(path_posts_templ) == len(path_segs_templ)
        
        all_segs_templ = [] # segmentations for this particular template from all models
        all_confid_templ = [] # confidence map for this particular template from all models

        save_vol, save_shp, save_aff, save_n_dims, save_n_channels, save_h, save_im_res = get_volume_info(path_segs_templ[0], return_volume=True, aff_ref=None, max_channels=10)
        print(f"template #{str(templ_ind)}")

        for model_ind in range(len(path_posts_templ)): # for each model

            print("seg " + str(path_segs_templ[model_ind].split('/')[-2]) + "/" + str(path_segs_templ[model_ind].split('/')[-1]) + " + post " + str(path_posts_templ[model_ind].split('/')[-2]) + "/" + str(path_posts_templ[model_ind].split('/')[-1]))
            seg_vol, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(path_segs_templ[model_ind], return_volume=True, aff_ref=None, max_channels=10)
            post_vol, post_shp, post_aff, post_n_dims, post_n_channels, post_h, post_im_res = get_volume_info(path_posts_templ[model_ind], return_volume=True, aff_ref=None, max_channels=10)
            assert (np.all(save_shp == seg_shp))
            assert (np.all(np.abs(save_aff - seg_aff) < 0.0001)), print(str(save_aff) + str(seg_aff))
            assert (save_n_dims == seg_n_dims)
            assert (save_n_channels == seg_n_channels)
            assert (np.all(save_im_res == seg_im_res))
            assert np.all(seg_vol >= 0)
            assert isinstance(seg_vol[0, 0, 0], np.float64), type(seg_vol[0, 0, 0])
            assert isinstance(seg_im_res[0], np.float32)

            confidence_vol = np.max(post_vol, axis=-1, out=None, keepdims=False)

            all_segs_templ.append(seg_vol)
            all_confid_templ.append(confidence_vol)

        all_segs_templ = np.stack(all_segs_templ) # shape: (#models, l, w, h)
        all_confid_templ = np.stack(all_confid_templ) # shape: (#models, l, w, h)

        inds_most_confid_templ = np.argmax(all_confid_templ, axis=0) # shape: (l, w, h)

        new_vol = all_segs_templ[inds_most_confid_templ, np.arange(inds_most_confid_templ.shape[0])[:, None, None], np.arange(inds_most_confid_templ.shape[1])[None, :, None], np.arange(inds_most_confid_templ.shape[2])[None, None, :]]

        save_path = os.path.join(path_merged, "merged_" + path_segs_templ[0].split('/')[-1])
        save_volume(volume=new_vol, aff=save_aff, header=save_h, path=save_path, res=save_im_res, dtype='float64', n_dims=save_n_dims)



# if __name__ == "__main__":
#     ensemble_combine_anyN(
#     path_posts=["/cluster/work/menze/zshang/data/CHUV/experiments/results/merged/synth1v1+dhcp_sep_noclst_e14_merge_r_exp7_e12_merge_r2_exp7_e20/post1", "/cluster/work/menze/zshang/data/CHUV/experiments/results/merged/synth1v1+dhcp_sep_noclst_e14_merge_r_exp7_e12_merge_r2_exp7_e20/post2", "/cluster/work/menze/zshang/data/CHUV/experiments/results/merged/synth1v1+dhcp_sep_noclst_e14_merge_r_exp7_e12_merge_r2_exp7_e20/post3"],
#     path_segs=["/cluster/work/menze/zshang/data/CHUV/experiments/results/merged/synth1v1+dhcp_sep_noclst_e14_merge_r_exp7_e12_merge_r2_exp7_e20/seg1", "/cluster/work/menze/zshang/data/CHUV/experiments/results/merged/synth1v1+dhcp_sep_noclst_e14_merge_r_exp7_e12_merge_r2_exp7_e20/seg2", "/cluster/work/menze/zshang/data/CHUV/experiments/results/merged/synth1v1+dhcp_sep_noclst_e14_merge_r_exp7_e12_merge_r2_exp7_e20/seg3"],
#     path_merged="/cluster/work/menze/zshang/data/CHUV/experiments/results/merged/synth1v1+dhcp_sep_noclst_e14_merge_r_exp7_e12_merge_r2_exp7_e20/raw_seg",
#     rule="max_posterior")
