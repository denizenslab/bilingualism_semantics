"""Plot semantic cluster flatmaps, % match between languages, tuning shift per cluster.

Note: Cluster numbers in the paper are flipped and 1-indexed.
Figure 3B, 3C, 5A.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from matplotlib.colors import ListedColormap
from scipy.stats import ttest_1samp
from sklearn.metrics import confusion_matrix

from utils import (
    cmap_dir,
    do_get_clusters_from_centroids,
    get_clusters_dicts,
    get_pssd,
    get_voxel_to_fsaverage_mapper,
    load_model_results,
    num_fsaverage_vertices,
    participant_names,
    save_flatmap_image,
    template_surface
)


broyg_cmap = ListedColormap(plt.imread(os.path.join(cmap_dir, "BROYG.png")).squeeze())
np.random.seed(123)
threshold = 0.1
score_type = "r2"
model_type = "fasttext"
num_permutation_iters = 1000
min_participants_for_group = 1

figures_dir = f"../results/figures/clusters_panels_{model_type}"
os.makedirs(figures_dir, exist_ok=True)

results_dict = load_model_results(model_type)
if model_type == "fasttext":
    do_flip_colors = True
    do_flip_clusters = True
elif model_type == "mBERT":
    do_flip_colors = False
    do_flip_clusters = True

n_clusters = 5
cluster_colors = ["#3a3f99", "#eb0012", "#ff9227", "#f5ef00", "#17b530"]
all_primal_weights_dict = results_dict["all_primal_weights_dict"]
all_scores_dict = results_dict["all_scores_dict"]
weights_diff_scaled_dict = results_dict["weights_diff_scaled_dict"]

def significance_test_match(clusters_en, clusters_zh, num_permutation_iters=1000):
    num_gt_match = 0
    match_percent = (clusters_en == clusters_zh).mean()
    for _ in range(num_permutation_iters):
        np.random.shuffle(clusters_en)
        permuted_match = (clusters_en == clusters_zh).mean()
        if permuted_match >= match_percent:
            num_gt_match += 1
    return num_gt_match / num_permutation_iters

clusters_dict_en, clusters_dict_zh = get_clusters_dicts(model_type)

num_participants_well_pred = np.zeros(num_fsaverage_vertices)
for participant_idx, participant in enumerate(participant_names):
    scores_en = all_scores_dict["en"][participant]
    scores_zh = all_scores_dict["zh"][participant]
    included_voxels = np.where((scores_en > threshold) & (scores_zh > threshold))
    excluded_voxels = list(
        set(range(np.array(scores_en.size))) - set(included_voxels[0])
    )
    included_voxels_mask = np.zeros(scores_en.size) * np.nan
    included_voxels_mask[included_voxels] = 1
    voxel_to_fsaverage_mapper = get_voxel_to_fsaverage_mapper(participant)
    fsaverage_mask = voxel_to_fsaverage_mapper @ np.nan_to_num(included_voxels_mask)
    fsaverage_mask[fsaverage_mask > 0] = 1 
    num_participants_well_pred += fsaverage_mask

    clusters_en = np.copy(clusters_dict_en[participant].astype(float))
    clusters_en[clusters_en == n_clusters + 1] = np.nan
    clusters_en[excluded_voxels] = np.nan
    clusters_zh = np.copy(clusters_dict_zh[participant].astype(float))
    clusters_zh[clusters_zh == n_clusters + 1] = np.nan
    clusters_zh[excluded_voxels] = np.nan

    fig_flatmap, ax_flatmap = plt.subplots()
    
    # Plot confusion matrix: Figure 3C.
    non_nan_voxels = np.where(~np.isnan(clusters_en))
    clusters_en_non_nan = clusters_en[non_nan_voxels]
    clusters_zh_non_nan = clusters_zh[non_nan_voxels]
    confusion_mat = confusion_matrix(
        clusters_en_non_nan,
        clusters_zh_non_nan,
        labels=range(n_clusters),
    )
    match_percent = (clusters_en_non_nan == clusters_zh_non_nan).mean()
    
    # Significance test.
    num_gt_match = 0
    for iter_num in range(num_permutation_iters):
        np.random.shuffle(clusters_en_non_nan)
        permuted_match = (clusters_en_non_nan == clusters_zh_non_nan).mean()
        if permuted_match >= match_percent:
            num_gt_match += 1
    print(
        n_clusters,
        participant,
        "match %",
        match_percent,
        "match pval",
        num_gt_match / num_permutation_iters,
    )
    confusion_mat = confusion_mat / confusion_mat.sum()
    if do_flip_clusters:
        confusion_mat = np.flip(confusion_mat, axis=1)
        confusion_mat = np.flip(confusion_mat, axis=0)
    plt.imshow(confusion_mat, cmap="inferno", vmin=0, vmax=0.8)
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat)):
            _ = plt.text(
                j,
                i,
                np.round(confusion_mat[i, j], 2),
                ha="center",
                va="center",
                color="w",
            )
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Chinese Cluster Assignment")
    plt.ylabel("English Cluster Assignment")
    plt.title(participant)
    plt.savefig(
        os.path.join(
            figures_dir,
            f"confusion_matrix_{model_type}_clusters_{participant}.png",
        ),
    )
    plt.close()

    # Plot flatmaps.
    for language in ["en", "zh"]:
        if language == "en":
            clusters_to_plot = clusters_en
        elif language == "zh":
            clusters_to_plot = clusters_zh
        save_flatmap_image(clusters_to_plot, participant, save_path=os.path.join(figures_dir, f"{participant}_{language}_{model_type}"),
                       cmap=broyg_cmap, vmin=-0.2, vmax=4.3) 

# Plot histograms per cluster. Figure 5A.
bins = np.arange(-1, 1, 0.05)

pssd = get_pssd(model_type).squeeze()
if do_flip_colors:
    pssd = -pssd

for language in ["en", "zh"]:
    fig, axes = plt.subplots(1, n_clusters, figsize=(9.3, 1.3))
    for participant_idx, participant in enumerate(participant_names):
        scores_en = all_scores_dict["en"][participant]
        scores_zh = all_scores_dict["zh"][participant]
        included_voxels = np.where(
            (scores_en > threshold) & (scores_zh > threshold)
        )
        excluded_voxels = list(
            set(range(np.array(scores_en.size))) - set(included_voxels[0])
        )

        clusters_en = np.copy(
            clusters_dict_en[participant].astype(float)
        )
        clusters_en[excluded_voxels] = np.nan

        clusters_zh = np.copy(
            clusters_dict_zh[participant].astype(float)
        )
        clusters_zh[excluded_voxels] = np.nan

        proj_diff = np.zeros(scores_en.size) * np.nan
        weights_diff = weights_diff_scaled_dict[participant]
        proj_diff[included_voxels] = (
            pssd @ weights_diff / np.linalg.norm(weights_diff, axis=0)
        )[included_voxels]

        for cluster_idx, color in enumerate(cluster_colors):
            if do_flip_clusters:
                ax_idx = int(n_clusters - cluster_idx - 1)
            else:
                ax_idx = int(cluster_idx)
            ax = axes[ax_idx]
            if language == "en":
                cluster_voxels = np.where(clusters_en == cluster_idx)
            elif language == "zh":
                cluster_voxels = np.where(clusters_zh == cluster_idx)
            elif language == "both":
                cluster_voxels = np.where(
                    (clusters_en == cluster_idx) & (clusters_zh == cluster_idx)
                )
            mean_proj = np.nanmean(proj_diff[cluster_voxels])
            ax.axvline(0, color="k", alpha=1, linestyle="--")
            ax.hist(
                proj_diff[cluster_voxels],
                bins=bins,
                alpha=1,
                color=color,
                density=True,
            )
            # remove frame
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            ax.set_xlim([-1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            non_nan_cluster_voxels = list(
                set(list(np.where(~np.isnan(proj_diff))[0])).intersection(
                    list(cluster_voxels[0])
                )
            )
            ttest_result = ttest_1samp(
                np.arctanh(proj_diff[cluster_voxels]), 0  # , alternative=alternative
            )
            print(
                participant,
                language,
                cluster_idx,
                "stat",
                round(ttest_result.statistic, 2),
                "pvalue",
                round(ttest_result.pvalue, 5),
            )
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                figures_dir,
                f"{model_type}_{n_clusters}_clusters_{participant}_{language}_histograms.png",
            ),
        )
        plt.close(fig)

# Plot for group.
group_weights_en = np.zeros((all_primal_weights_dict['en'][participant_names[0]].shape[0], num_fsaverage_vertices))
group_weights_zh = np.zeros((all_primal_weights_dict['en'][participant_names[0]].shape[0], num_fsaverage_vertices))
for participant_idx, participant in enumerate(participant_names):
    voxel_to_fsaverage_mapper = get_voxel_to_fsaverage_mapper(participant)
    group_weights_en += (all_primal_weights_dict['en'][participant] / np.linalg.norm(all_primal_weights_dict['en'][participant], axis=0)) @ voxel_to_fsaverage_mapper.T
    group_weights_zh += (all_primal_weights_dict['zh'][participant] / np.linalg.norm(all_primal_weights_dict['zh'][participant], axis=0)) @ voxel_to_fsaverage_mapper.T
group_weights_en /= len(participant_names)
group_weights_zh /= len(participant_names)
group_weights_en /= np.linalg.norm(group_weights_en, axis=0)
group_weights_zh /= np.linalg.norm(group_weights_zh, axis=0)

# Get cluster assignments
not_all_nan_mask = np.zeros(num_fsaverage_vertices)
not_all_nan_mask[
    np.where(num_participants_well_pred >= min_participants_for_group)
] = 1
excluded_vertices = np.where(not_all_nan_mask == 0)
clusters_en = do_get_clusters_from_centroids(group_weights_en.T, clusters_dict_en["cluster_avgs"]).astype(float)
clusters_en[excluded_vertices] = np.nan
clusters_zh = do_get_clusters_from_centroids(group_weights_zh.T, clusters_dict_zh["cluster_avgs"]).astype(float)
clusters_zh[excluded_vertices] = np.nan
for language in ["en", "zh"]:
    clusters_to_plot = {
        "en": clusters_en,
        "zh": clusters_zh,
    }[language]
    save_flatmap_image(clusters_to_plot, template_surface, save_path=os.path.join(figures_dir, f"group_{language}_{model_type}"),
                    cmap=broyg_cmap, vmin=-0.2, vmax=4.3) 

non_nan_voxels = np.where(~np.isnan(clusters_en))
clusters_en_non_nan = clusters_en[non_nan_voxels]
clusters_zh_non_nan = clusters_zh[non_nan_voxels]
confusion_mat = confusion_matrix(
    clusters_en_non_nan,
    clusters_zh_non_nan,
    labels=range(n_clusters),
)
match_percent = (clusters_en_non_nan == clusters_zh_non_nan).mean()
# Significance test.
print(
    "group match %",
    match_percent,
    "group match pval",
    num_gt_match / num_permutation_iters,
)
confusion_mat = confusion_mat / confusion_mat.sum()
if do_flip_clusters:
    confusion_mat = np.flip(confusion_mat, axis=1)
    confusion_mat = np.flip(confusion_mat, axis=0)
plt.imshow(confusion_mat, cmap="inferno", vmin=0, vmax=0.25)
for i in range(len(confusion_mat)):
    for j in range(len(confusion_mat)):
        _ = plt.text(
            j,
            i,
            np.round(confusion_mat[i, j], 2),
            ha="center",
            va="center",
            color="w",
        )
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.xlabel("Chinese Cluster Assignment")
plt.ylabel("English Cluster Assignment")
plt.title("Group")
plt.savefig(
    os.path.join(
        figures_dir,
        f"confusion_matrix_{model_type}_{n_clusters}_clusters_group.png",
    ),
)
plt.close()
