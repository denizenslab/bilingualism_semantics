import h5py
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

data_dir = "../data"
cmap_dir = "../data"
mappers_dir = os.path.join(data_dir, "mappers")
train_participants = ["p01", "p02", "p03", "p04"]
test_participants = ["p05", "p06"]
participant_names = train_participants + test_participants
languages = ["en", "zh"]
template_surface = "fsaverage"
num_fsaverage_vertices = 327684
scores_threshold = 0.1
full_group_pcs_filepath = "../results/outputs/pcs_diffs_{model_type}.p"
os.makedirs("../results/outputs", exist_ok=True)
color_en = "blue"
color_zh = "red"

participant_to_marker_dict = {
    "p01": "o",
    "p02": "s",
    "p03": "D",
    "p04": "^",
    "p05": "P",
    "p06": "X",
}

do_flip_colors_dict = {"fasttext": True, "fasttext_sig_mask": False, "mBERT": False}
regions_to_rois_dict = {
    "temporal": ["bankssts", "inferiortemporal", "middletemporal", "superiortemporal", "temporalpole", "transversetemporal", "fusiform", "entorhinal", "parahippocampal"],
    "parietal": ["inferiorparietal", "superiorparietal", "supramarginal", "precuneus", "isthmuscingulate", "posteriorcingulate"],
    "prefrontal": ["caudalmiddlefrontal", "parsopercularis", "parsorbitalis", "parstriangularis", "rostralmiddlefrontal", "superiorfrontal", "frontalpole",
                  "caudalanteriorcingulate"],
}


def get_bh_excluded_voxels(pvalues: np.ndarray,
        alpha: float = 0.05):
    """Returns indices of voxels that are not significant after Benjamini-Hochberg FDR correction.

    Parameters:
    -----------
    pvalues : np.ndarray : 1D array of p-values for each voxel.
    alpha : float : significance threshold.

    Returns:
    --------
    voxels_excluded : np.ndarray : 1D array of indices of voxels that are not significant after FDR correction.
    """
    num_values = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    max_p = pvalues_sorted[np.argmax(np.where(pvalues_sorted <= ((np.arange(1, num_values + 1) / num_values) * alpha)))]
    voxels_excluded = np.where(pvalues > max_p)[0]
    return voxels_excluded


def scale_weights_by_score(primal_weights, scores):
    """Scales weights by prediction accuracy scores.
    This is done in order to upweight well-predicted voxels and downweight poorly-predicted voxels.

    Parameters:
    -----------
    primal_weights : np.ndarray : num_feature_dims x num_voxels matrix of voxelwise model weights.
    scores : np.ndarray : num_voxels array of prediction accuracy scores.

    Returns:
    ---------
    scaled_weights : np.ndarray : num_feature_dims x num_voxels matrix of scaled voxelwise model weights.
    """
    primal_weights = np.copy(primal_weights)
    norm = np.linalg.norm(primal_weights, axis=0)
    primal_weights[:, norm != 0] /= norm[norm != 0]
    primal_weights *= np.maximum(0, scores)
    return np.nan_to_num(primal_weights)


def load_model_results(model_name):
    """Loads model results for all participants and languages.

    Parameters:
    -----------
    model_name : str : name of semantic model. Options: ["fasttext", "mBERT"]

    Returns:
    --------
    Dictionaries containing model results for all participants and languages.
    """
    scores_filepath_template = os.path.join(data_dir, "estimated_models", f"estimated_models_{model_name}", "scores", "{participant_name}_train_{train_language}_test_{test_language}_scores_pvalues.npz")
    primal_weights_filepath_template = os.path.join(data_dir, "estimated_models", f"estimated_models_{model_name}", "weights", "{participant_name}_{language}_weights.npz")

    all_weights_diff_scaled_dict = dict()
    all_primal_weights_dict = {language: dict() for language in languages}
    all_scores_dict = {language: dict() for language in languages}
    all_across_scores_dict = {language: dict() for language in languages}
    all_across_excluded_voxels_dict = {language: dict() for language in languages}
    all_excluded_voxels_dict = {language: dict() for language in languages}
    all_primal_weights_scaled_dict = {
        language: dict() for language in languages
    }

    for participant in participant_names:
        for language, test_language in zip(languages, languages[::-1]):
            # Across language
            across_scores_pvalues_dict = np.load(
                scores_filepath_template.format(
                    participant_name=participant,
                    train_language=language,
                    test_language=test_language,
                )
            )
            all_across_scores_dict[language][participant] = across_scores_pvalues_dict["scores_sqrt_r2"]
            across_excluded_voxels = get_bh_excluded_voxels(across_scores_pvalues_dict["pvalues"])
            all_across_excluded_voxels_dict[language][
                participant
            ] = across_excluded_voxels

            # Within language
            scores_pvalues_dict = np.load(
                scores_filepath_template.format(
                    participant_name=participant,
                    train_language=language,
                    test_language=language,
                )
            )
            all_scores_dict[language][participant] = scores_pvalues_dict["scores_sqrt_r2"]
            excluded_voxels = get_bh_excluded_voxels(scores_pvalues_dict["pvalues"])
            all_excluded_voxels_dict[language][participant] = excluded_voxels

            all_primal_weights_dict[language][participant] = np.load(primal_weights_filepath_template.format(participant_name=participant, language=language))["weights"]
        
        # scale by sqrt mean R2 score across langs (mean sqrt R2 gives qualitatively similar results)
        mean_scores_across_langs = np.nan_to_num(
            np.sqrt(
                (
                    all_scores_dict["en"][participant] ** 2
                    + all_scores_dict["zh"][participant] ** 2
                )
                / 2
            )
        )
        weights_en_scaled = scale_weights_by_score(
            all_primal_weights_dict["en"][participant],
            mean_scores_across_langs,
        )
        weights_zh_scaled = scale_weights_by_score(
            all_primal_weights_dict["zh"][participant], mean_scores_across_langs
        )
        weights_diff = (
            weights_en_scaled - weights_zh_scaled
        )  # (n_feature_dims, n_voxels)

        all_weights_diff_scaled_dict[participant] = weights_diff
        all_primal_weights_scaled_dict["en"][participant] = weights_en_scaled
        all_primal_weights_scaled_dict["zh"][participant] = weights_zh_scaled

    return {
        "all_primal_weights_dict": all_primal_weights_dict,
        "all_scores_dict": all_scores_dict,
        "all_across_scores_dict": all_across_scores_dict,
        "weights_diff_scaled_dict": all_weights_diff_scaled_dict,
        "all_primal_weights_scaled_dict": all_primal_weights_scaled_dict,
        "all_excluded_voxels_dict": all_excluded_voxels_dict,
        "all_across_excluded_voxels_dict": all_across_excluded_voxels_dict,
    }


def get_stimulus_word_embeddings(model_name: str):
    """Loads embeddings of stimulus words for a given model.
    
    Parameters:
    -----------
    model_name : str : name of semantic model. Options: ["fasttext", "mBERT"]

    Returns:
    --------
    word_embedding_matrix_en : np.ndarray : num_words x num_dims matrix of English word embeddings.
    word_embedding_matrix_zh : np.ndarray : num_words x num_dims matrix of Chinese word embeddings.
    words_list_en : list : list of English words.
    words_list_zh : list : list of Chinese words.
    """
    embeddings_dict = np.load(os.path.join(data_dir, "stimuli", f"{model_name}_stimulus_embeddings.npz"))
    word_embedding_matrix_en = embeddings_dict["word_embedding_matrix_en"]
    word_embedding_matrix_zh = embeddings_dict["word_embedding_matrix_zh"]
    words_list_en = embeddings_dict["words_list_en"]
    words_list_zh = embeddings_dict["words_list_zh"]
    return (
        word_embedding_matrix_en,
        word_embedding_matrix_zh,
        words_list_en,
        words_list_zh,
    )


def compute_variance_explained_ratio(matrix, pcs):
    """Returns the variance explained by each principal component.

    Parameters:
    -----------
    matrix : np.ndarray : num_dims x num_samples matrix. 
    pcs : np.ndarray : num_pcs x num_dims matrix of principal components.

    Returns:
    --------
    variance_explained_ratio : np.ndarray : num_pcs array of variance explained by each principal component.
    """
    pc_projections = np.dot(pcs, matrix)  # num_pcs x num_samples
    var_per_pc = np.nanvar(pc_projections, axis=1)
    original_variance = np.nansum(np.nanvar(matrix, axis=1))
    return var_per_pc / original_variance


def get_voxels_for_region(participant_id):
    """Returns indices of voxels for each region.

    Parameters:
    -----------
    participant_id : str : participant ID.

    Returns:
    --------
    region_to_voxels_dict : dict : dictionary that maps region names to indices of voxels.
    """
    roi_to_voxels = np.load(os.path.join(mappers_dir, f"{participant_id}_roi_to_voxels.npz"))
    num_voxels = roi_to_voxels["num_voxels"]
    region_to_voxels_dict = dict()
    for hemi_name in ["lh", "rh"]:
        for region_name, rois in regions_to_rois_dict.items():
            region_mask = np.zeros(num_voxels) * np.nan
            for roi in rois:
                region_mask[np.array(roi_to_voxels[roi + hemi_name])] = 1
            region_voxels = np.where(region_mask == 1)[0]
            region_to_voxels_dict[region_name + hemi_name] = region_voxels
    return region_to_voxels_dict


def get_voxel_to_fsaverage_mapper(participant_name: str):
    """Returns the mapper from native space to fsaverage.

    Parameters:
    -----------
    participant_name : str : participant ID.

    Returns:
    --------
    mapper : scipy.sparse.csr_matrix : sparse matrix that maps voxels from native space to fsaverage.
    """
    mapper = load_hdf5_sparse_array(
        os.path.join(mappers_dir, f"{participant_name}_mapper.hdf"),
        key="voxel_to_fsaverage",
    )
    return mapper


def load_sparse_array(fname, varname):
    """Load a numpy sparse array from an hdf file

    Parameters
    ----------
    fname: string
        file name containing array to be loaded
    varname: string
        name of variable to be loaded

    Notes
    -----
    This function relies on variables being stored with specific naming
    conventions, so cannot be used to load arbitrary sparse arrays.

    By Mark Lescroart

    """
    with h5py.File(fname) as hf:
        data = (hf['%s_data'%varname], hf['%s_indices'%varname], hf['%s_indptr'%varname])
        sparsemat = scipy.sparse.csr_matrix(data, shape=hf['%s_shape'%varname])
    return sparsemat


def save_flatmap_image(values, participant_name, save_path, plot_nan_as_gray=True, **kwargs):
    """Plot flatmap image from data array

    Parameters:
    -----------
    values : array : n_voxels or n_vertices array of voxel or vertex values to be mapped.
    participant_name : str : name of surface to map to.
    save_path : str : path to save the flatmap image.
    plot_nan_as_gray : bool : whether to plot NaN values as gray (mimic curvature without surface data).
    **kwargs : dict : additional keyword arguments to pass to plotting function.
    """
    flatmap = map_to_flat(values, participant_name)
    if plot_nan_as_gray:
        values_for_curvature = np.zeros(values.shape)
        values_for_curvature[np.isnan(values)] = 1
        flatmap_nan = map_to_flat(values_for_curvature, participant_name)
        _ = plt.imshow(flatmap_nan, interpolation="nearest", cmap="Greys", vmin=-1, vmax=3, zorder=0)
    _ = plt.imshow(flatmap, interpolation="nearest", zorder=1, **kwargs)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


def map_to_flat(values, participant_name):
    """Generate flatmap image for an individual subject from voxel array

    This function maps a list of voxels into a flattened representation
    of an individual subject's brain.

    Parameters
    ----------
    values: array
        n x 1 array of voxel values to be mapped
    mapper_file: string
        file containing mapping arrays

    Returns
    -------
    image : array
        flatmap image, (n x 1024)

    By Mark Lescroart

    """
    mapper_file = os.path.join(mappers_dir, f"{participant_name}_mapper.hdf")
    pixmap = load_sparse_array(mapper_file, 'pixmap')
    with h5py.File(mapper_file, mode='r') as hf:
        pixmask = hf['pixmask'][()]
    badmask = np.array(pixmap.sum(1) > 0).ravel()
    img = (np.nan * np.ones(pixmask.shape)).astype(values.dtype)
    mimg = (np.nan * np.ones(badmask.shape)).astype(values.dtype)
    # nanmean
    averaged_data = pixmap.dot(np.nan_to_num(values.ravel()))
    ignored = np.isnan(values.ravel())
    weights_not_ignored = pixmap.dot((~ignored).astype(values.dtype))
    averaged_data /= weights_not_ignored
    mimg[badmask] = averaged_data[badmask].astype(mimg.dtype)
    img[pixmask] = mimg
    return img.T[::-1]


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def draw_violinplot(data, positions, color, widths=0.1, alpha=0.3):
    parts = plt.violinplot(
        data, positions, showmeans=False, showmedians=False,
        showextrema=False, widths=widths * 1.9)

    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(alpha)


def get_clusters_dicts(model_type: str):
    """Returns dictionaries of clusters for each language.

    Parameters:
    -----------
    model_type : str : name of semantic model. Options: ["fasttext", "mBERT"]

    Returns:
    --------
    clusters_dict_en : dict : dictionary of clusters for English.
    clusters_dict_zh : dict : dictionary of clusters for Chinese.
    """
    clusters_dict_en = load_pickle(
        os.path.join(data_dir, "estimated_models", f"estimated_models_{model_type}", "semantic_clusters", "clusters_en.p")
    )
    clusters_dict_zh = load_pickle(
        os.path.join(data_dir, "estimated_models", f"estimated_models_{model_type}", "semantic_clusters", "clusters_zh.p")
    )
    return clusters_dict_en, clusters_dict_zh


def get_vector_top_bottom_word_projections(
    vectors, words_list, word_embedding_matrix, num_projected_words=20
):
    """Get words with the highest and lowest projections onto a set of vectors.

    Parameters:
    -----------
    vectors : np.ndarray
        num_vectors x num_dims matrix of vectors to project word embeddings onto
    words_list : list
        [num_words] length list of words
    word_embedding_matrix : np.ndarray
        num_words x num_dims matrix of word embeddings corresponding to words_list
    """
    word_embedding_matrix = word_embedding_matrix / np.linalg.norm(
        word_embedding_matrix, axis=1
    ).reshape(-1, 1)
    word_projections = np.dot(vectors, word_embedding_matrix.T)  # num_vectors x num_words
    word_projections_dict = dict()
    for pc_index, pc_word_projections in enumerate(word_projections):
        sorted_word_indices = np.argsort(
            np.nan_to_num(pc_word_projections)
        )  # most negative vals at lowest indices.
        top_projections = words_list[sorted_word_indices][-num_projected_words:][::-1]
        bottom_projections = words_list[sorted_word_indices][:num_projected_words]
        words_list_sorted = words_list[sorted_word_indices]
        word_projections_dict[pc_index] = {
            "words": words_list_sorted,
            "projections": pc_word_projections[sorted_word_indices],
            "top": top_projections,
            "bottom": bottom_projections,
        }
    return word_projections_dict


def get_closest_farthest_words(
    vectors,
    words_list,
    word_embedding_matrix,
    threshold=123,
    num_projected_words: int = 30,
):
    word_embedding_matrix = word_embedding_matrix / np.linalg.norm(
        word_embedding_matrix, axis=1
    ).reshape(-1, 1)
    word_projections = np.dot(vectors, word_embedding_matrix.T)  # num_vectors x num_words

    word_projections_dict = dict()
    for vector_idx, vector_word_projections in enumerate(word_projections):
        sorted_word_indices = np.argsort(np.nan_to_num(vector_word_projections))
        sorted_projections = vector_word_projections[sorted_word_indices]
        sorted_words = words_list[sorted_word_indices]
        indices_subset = [
            idx
            for idx, word in enumerate(sorted_words)
            if np.count_nonzero(words_list == word) < threshold
        ]  # NOTE: threshold corresponds to top 1% of words.
        word_indices = np.concatenate(
            [np.arange(num_projected_words)]
        )
        sorted_projections_neg = np.array(
            list(dict.fromkeys(list(sorted_projections[indices_subset])).keys())
        )[word_indices]
        sorted_words_neg = np.array(
            list(dict.fromkeys(list(sorted_words[indices_subset])).keys())
        )[word_indices]

        sorted_projections_pos = np.array(
            list(dict.fromkeys(list(sorted_projections[indices_subset][::-1])).keys())
        )[word_indices]
        sorted_words_pos = np.array(
            list(dict.fromkeys(list(sorted_words[indices_subset][::-1])).keys())
        )[word_indices]

        word_projections_dict[vector_idx] = {
            "sorted_projections_neg": sorted_projections_neg,
            "sorted_words_neg": sorted_words_neg,
            "sorted_projections_pos": sorted_projections_pos,
            "sorted_words_pos": sorted_words_pos,
        }
    return word_projections_dict


def get_pssd(model_type):
    """Load the PSSD. Re-compute if not already cached."""
    pssd_filepath = full_group_pcs_filepath.format(model_type=model_type)
    if os.path.exists(pssd_filepath):
        with open(pssd_filepath, "rb") as f:
            pssd = pickle.load(f)["pssd"]
    else:
        results_dict = load_model_results(model_type)

        all_scores_dict = results_dict["all_scores_dict"]
        weights_diff_mean_lang_dict = results_dict["weights_diff_scaled_dict"]
        weights_diff_group = []
        for participant, weights_diff in weights_diff_mean_lang_dict.items():
            scores_en = all_scores_dict["en"][participant]
            scores_zh = all_scores_dict["zh"][participant]
            included_voxels = np.where((scores_en > scores_threshold) & (scores_zh > scores_threshold))
            weights_diff_group.append(weights_diff[:, included_voxels].squeeze())
        weights_diff_group_cat = np.concatenate(weights_diff_group, axis=1)
        tuning_shift_pca = PCA(n_components=1)
        tuning_shift_pca.fit(weights_diff_group_cat.T)
        pssd = tuning_shift_pca.components_
        with open(pssd_filepath, "wb") as f:
            pickle.dump({"pssd": pssd}, f)
    return pssd.reshape(1, -1)


def load_hdf5_sparse_array(file_name, key):
    """Load a scipy sparse array from an hdf file

    Parameters
    ----------
    file_name : string
        File name containing array to be loaded.
    key : string
        Name of variable to be loaded.

    Notes
    -----
    This function relies on variables being stored with specific naming
    conventions, so cannot be used to load arbitrary sparse arrays.

    Taken from voxelwise_tutorials (https://github.com/gallantlab/voxelwise_tutorials.git).
    """
    OLD_KEYS = {
        "flatmap_mask": "pixmask",
        "voxel_to_flatmap": "pixmap",
    }
    with h5py.File(file_name, mode='r') as hf:

        # Some keys have been renamed. Use old key on KeyError.
        if '%s_data' % key not in hf.keys() and key in OLD_KEYS:
            key = OLD_KEYS[key]

        # The voxel_to_fsaverage mapper is sometimes split between left/right.
        if (key == "voxel_to_fsaverage" and '%s_data' % key not in hf.keys()
                and "vox_to_fsavg_left_data" in hf.keys()):
            left = load_hdf5_sparse_array(file_name, "vox_to_fsavg_left")
            right = load_hdf5_sparse_array(file_name, "vox_to_fsavg_right")
            return scipy.sparse.vstack([left, right])

        data = (hf['%s_data' % key], hf['%s_indices' % key],
                hf['%s_indptr' % key])
        sparsemat = scipy.sparse.csr_matrix(data, shape=hf['%s_shape' % key])
    return sparsemat

def do_get_clusters_from_centroids(weight_data, cluster_centroids, dist_metric='euclidean'):
    """ Clusters weights according to pre-determined cluster centroids.
    
    Parameters
    ----------
    weight_data : array of shape (n_targets, n_features)
    cluster_centroids : array of shape (n_clusters, n_features)
    dist_metric : str
        Type of distance to cluster targets by.
        See scipy.spatial.distance.cdist for all options.
    
    Returns
    -------
    clustering : (n_targets)
    """
    distances = cdist(cluster_centroids, weight_data, metric=dist_metric)  # num_centroids x num_targets
    clustering = distances.argmin(0)
    return clustering
