
import nibabel as nib
import glob
import os
import numpy as np
from ext.lab2im.utils import get_volume_info, save_volume, get_list_labels, list_images_in_folder
from scipy.signal import convolve
from skimage import measure
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def process_feratures( 
    raw_features=None, # the raw features, shape (Nsegs, 21)
    n_clusters=3, # number of groups to divide the dhcp dataset  
    clustering_method='gmm', # clustering method: either "gmm" or "kmeans"
    n_components=5, # dimensions to reduce to by using PCA
    save_plot=False, # whether or not to save the plotted clusters 
    accord_exp=False, # whether or not to boost the importance of the chosen features
    fig_dir=None): # where to save the cluster plots

    # print(raw_features.shape) must be [n_samples, n_features]
    normalized_data = MinMaxScaler().fit_transform(raw_features)
    assert normalized_data.shape == (raw_features.shape[0], 21)

    if accord_exp:
        print("normalized_data[:, [0,1,2,3,4,5,9,10,11,15,16,17]] *= 2")
        normalized_data[:, [0,1,2,3,4,5,9,10,11,15,16,17]] *= 2

    # dimentionality reduction with PCA 
    assert normalized_data.shape == (raw_features.shape[0], 21)
    pca = PCA(n_components=n_components)
    pca.fit(normalized_data) # (n_samples, n_features)
    lowdim_features = pca.transform(normalized_data) # (n_samples, n_features)
    assert lowdim_features.shape == (raw_features.shape[0], n_components)
    # print("np.std(lowdim_features, axis=0)")
    # print(np.std(lowdim_features, axis=0))
    # print("pca.components_")
    # print(pca.components_)
    print("pca.explained_variance_ratio_")
    print(pca.explained_variance_ratio_)

    # k-means clustering
    if clustering_method=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=10)
        kmeans.fit(lowdim_features) #(n_samples, n_features)
        clusters = kmeans.labels_
        centers = kmeans.cluster_centers_

        assert np.all(clusters == kmeans.fit_predict(lowdim_features))
        assert np.all(np.abs(centers - kmeans.cluster_centers_) < 0.0001)
        assert centers.shape[1] == lowdim_features.shape[1]

        # closest_inds, _ = pairwise_distances_argmin_min(centers, lowdim_features)
        # print("closest subject index for each cluster centroid (note that these indices start from zero)" + str(closest_inds))
        # closest_inds = np.array(closest_inds)
        # print(np.array(sorted(glob.glob(seg_path + '/*')))[closest_inds])
        # print("clusters=")
        # print(clusters)
        print("kmeans.inertia_")
        print(kmeans.inertia_)

    if clustering_method == 'gmm':
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(lowdim_features) # (n_samples, n_features)
        clusters = gmm.predict(lowdim_features) # (n_samples, n_features)
        centers = gmm.means_

        assert centers.shape[1] == lowdim_features.shape[1]
        # closest_inds, _ = pairwise_distances_argmin_min(centers, lowdim_features)
        # print("closest subject index for each cluster centroid (note that these indices start from zero)" +str(closest_inds))
        # print("clusters=")
        # print(clusters)
    
    assert len(clusters) == lowdim_features.shape[0]

    if save_plot:
        first_two = lowdim_features[:,:2]
        colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4:'yellow', 5:'pink', 6:'black', 7:'gray', 8:'orange', 9:'brown'}
        color_map = np.array([colors[cls] for cls in list(clusters)])
        plt.figure(figsize=(8, 6))
        plt.scatter(first_two[:, 0], first_two[:, 1], edgecolors='w', linewidth=0.5, c=color_map, alpha=0.6)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Scatter Plot of Data Points')
        plt.grid(True)
        plt.savefig(fig_dir + 'img_10_cluster_3_compo_acexp_01234591011151617_gmm.png')

    return clusters


def extract_pret( 
    seg_path = None, # path to folder containing label maps 
    inner_labels = None, # list of all labels to consider 
    n_clusters_dhcp=8, # number of groups to divide the dhcp dataset  
    n_clusters_zurich=6, # number of groups to divide the zurich dataset 
    clustering_method='gmm', # clustering method: either "gmm" or "kmeans"
    n_components=3, # dimensions to reduce to by using PCA
    save_plot=False, # Whether or not to plot the clusters 
    accord_exp=False, # whether or not to boost the importance of the chosen features
    load = False, # whether or not to load the existing raw features from param "saved_features_path"
    saved_features_path=None, # where the raw features are saved/should be saved
    path_to_save_weights=None, # where the final weights are saved 
    fig_dir=None, # where to save the cluster plots
    first_dhcp_ind=0): # the index of the first subject in the dhcp dataset 

    print("extract()")
    seg_list_all = np.array(sorted(glob.glob(seg_path + '/*')))

    if not load:

        assert inner_labels is not None

        all_features = np.zeros(21, dtype=float)
        for i in range(len(seg_list_all)):
            print("processing: " + str(seg_list_all[i].split('/')[-1].split('_')[0]))
            seg_vol, seg_shp, seg_aff, seg_n_dims, seg_n_channels, seg_h, seg_im_res = get_volume_info(seg_list_all[i], return_volume=True, aff_ref=None, max_channels=10)

            assert isinstance(seg_vol[0, 0, 0], np.float64)
            assert isinstance(seg_im_res[0], np.float32)

            inner_labels = np.array(inner_labels).astype(np.float64)
            mask_whole = seg_vol == inner_labels[0]
            for lbl in inner_labels[1:]:
                mask_whole = np.logical_or(mask_whole, seg_vol == lbl)

            total_volume = float(np.sum(mask_whole))
            verts, faces, _, _ = measure.marching_cubes(mask_whole, method='lewiner') 

            total_surface_area = float(measure.mesh_surface_area(verts, faces))
            assert (total_surface_area>0) and (total_volume>0)
            surface_to_volumn = total_surface_area / total_volume
            # print(total_surface_area)
            # print(total_volume)

            # all except bg and csf
            struct_vols = []
            struct_areas = []
            struct_surface_to_volumn = []
            for lb in inner_labels:
                mask_struct = seg_vol == lb
                verts, faces, _, _ = measure.marching_cubes(mask_struct, method='lewiner')
                struct_area = float(measure.mesh_surface_area(verts, faces))
                struct_vol = float(np.sum(mask_struct))
                struct_areas.append(struct_area)
                struct_vols.append(struct_vol)
                assert (struct_area>0) and (struct_vol>0)
                struct_surface_to_volumn.append(struct_area/struct_vol)
            struct_vols = np.array(struct_vols)
            struct_areas = np.array(struct_areas)
            struct_surface_to_volumn = np.array(struct_surface_to_volumn)
            struct_vols = struct_vols / np.sum(struct_vols)
            struct_areas = struct_areas / np.sum(struct_areas)

            assert (np.all(struct_vols<1) and np.all(struct_areas<1)) 
            assert len(np.array([total_volume])) == 1
            assert len(np.array([total_surface_area])) == 1
            assert len(np.array([surface_to_volumn])) == 1
            assert len(struct_vols) == 6
            assert len(struct_areas) == 6
            assert len(struct_surface_to_volumn) == 6

            vol_feature = np.concatenate((np.array([total_volume]), np.array([total_surface_area]),np.array([surface_to_volumn]), struct_vols, struct_areas, struct_surface_to_volumn))
            all_features = np.vstack([all_features, vol_feature])
            # print(vol_feature)

        raw_features = all_features[1:]
        print(f"saving raw features to {saved_features_path}...")
        np.save(saved_features_path,raw_features)
    else:
        print("loading raw features...")
        raw_features = np.load(saved_features_path)

    print("testing correctness of feature array...")
    assert raw_features.shape == (len(glob.glob(seg_path + '/*')), 21), "feature matrix has incorrect shape"
    # [(s_x/s_total)/(s_y/s_total)] / [(a_x/a_total)/(a_y/a_total)] = (s_x/a_x)/(s_y/a_y)
    test_assert = raw_features[:,9:14] / raw_features[:,10:15] / raw_features[:,3:8] * raw_features[:,4:9] - raw_features[:,15:20] / raw_features[:,16:]
    assert np.sum(np.abs(test_assert)) < 1e-10, "feature matrix is incorrectly calculated"
    assert np.sum(np.abs(raw_features[:,1] / raw_features[:,0] - raw_features[:,2])) < 1e-12, "feature matrix is incorrectly calculated"
    assert np.all(np.abs(np.sum(raw_features[:,3:9], axis=1) - 1) < 1e-12), "feature matrix is incorrectly calculated"
    assert np.all(np.abs(np.sum(raw_features[:,9:15], axis=1) - 1) < 1e-12), "feature matrix is incorrectly calculated"

    # generate mask for synth templates and dhcp templates: mask_synth, mask_dhcp, where path=synth, and neuro=dhcp
    mask_zurich = np.ones(len(seg_list_all)).astype(bool)
    mask_zurich[first_dhcp_ind:] = False 
    mask_dhcp = np.array([~m for m in mask_zurich])

    assert np.all(np.array([(("_rec-" in temppath) or ("_json_" in temppath)) for temppath in seg_list_all[mask_zurich]]))
    assert np.all(np.array(["_dseg_FeTA_labels.nii.gz" in temppath for temppath in seg_list_all[mask_dhcp]]))
    assert np.sum(mask_zurich) + np.sum(mask_dhcp) == len(raw_features)
    assert np.sum(mask_zurich) == 160
    assert np.sum(mask_dhcp) == 425 - 160


    print("processing dhcp")
    clusters_dhcp = process_feratures(    
        raw_features=raw_features[mask_dhcp, :],
        n_clusters=n_clusters_dhcp,
        clustering_method=clustering_method,
        n_components=n_components,
        save_plot=save_plot,
        accord_exp=accord_exp,
        fig_dir=fig_dir)
    
    print("processing synth")
    clusters_synth = process_feratures(    
        raw_features=raw_features[mask_zurich, :],
        n_clusters=n_clusters_zurich,
        clustering_method=clustering_method,
        n_components=n_components,
        save_plot=save_plot,
        accord_exp=accord_exp,
        fig_dir=fig_dir)

    # assign weights according to assigned cluster
    weight_per_cluster_dhcp = 0.5 / n_clusters_dhcp
    weight_per_cluster_synth = 0.5 / n_clusters_zurich
    counts_dhcp = np.bincount(clusters_dhcp)
    counts_synth = np.bincount(clusters_synth)
    print("number of templates in each cluster (dhcp): ")
    print(counts_dhcp)
    print("number of templates in each cluster (synth): ")
    print(counts_synth)

    weights = []
    # weight assigning example:
    # mask_path = [f,f,t,t,f,f,t,t,f,t]
    # mask_neuro= [t,t,f,f,t,t,f,f,t,f]
    # ind_path  = [0,0,0,1,2,2,2,3,4,4]
    # ind_neuro = [0,1,2,2,2,3,4,4,4,5]

    for ind in range(len(seg_list_all)):
        if not mask_zurich[ind]:
            ind_dhcp = np.sum(mask_dhcp[:ind])
            weights.append(weight_per_cluster_dhcp / counts_dhcp[clusters_dhcp[ind_dhcp]])
        else:
            ind_synth = np.sum(mask_zurich[:ind])
            weights.append(weight_per_cluster_synth / counts_synth[clusters_synth[ind_synth]])

    # validate weights: 
    print("validating weights...")
    for ass_ind in range(len(seg_list_all)):
        ass_weight = weights[ass_ind]
        if mask_dhcp[ass_ind]:
            ass_cluster = clusters_dhcp[np.sum(mask_dhcp[:ass_ind])]
            ass_count = np.sum(clusters_dhcp == ass_cluster)
            assert ass_weight == (0.5/n_clusters_dhcp/ass_count), ass_weight - (0.5/n_clusters_dhcp/ass_count)
        else:
            ass_cluster = clusters_synth[np.sum(mask_zurich[:ass_ind])]
            ass_count = np.sum(clusters_synth == ass_cluster)
            assert ass_weight == (0.5/n_clusters_zurich/ass_count), ass_weight - (0.5/n_clusters_zurich/ass_count)
    
    assert (np.sum(weights) >= 0.995) and (np.sum(weights) <= 1.00001), np.sum(weights)
    assert (np.sum(np.array(weights)[mask_zurich]) <= 0.50001) and (np.sum(np.array(weights)[mask_zurich]) >= 0.4975)

    if path_to_save_weights is not None:
        np.save(path_to_save_weights, weights)
        print(f"weights saved to: {path_to_save_weights}")

    return weights 

# how we extracted weights from training labal map sets containing both dHCP and zurich (synthetic) cases 
extract_pret(save_plot=False, seg_path='/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/synth1v1+dhcp', inner_labels=[2,3,4,5,6,7], n_clusters_dhcp=8, n_clusters_zurich=6, clustering_method='gmm', n_components=3, accord_exp=True, load=True, saved_features_path='/Users/ziyaoshang/Desktop/fa2023/SP/ziyao_aug2024/weights_features/synth1v1+dhcp_sep_clex1/features_synth1v1+dhcp_sep_clex1.npy', fig_dir=None, path_to_save_weights=None, first_dhcp_ind=160)

print('done')

