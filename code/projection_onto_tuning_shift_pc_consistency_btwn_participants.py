"""Plot consistency between participants in projection onto tuning shift PCs.

Figure 4C.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from matplotlib.patches import Patch
from scipy.stats import sem, ttest_1samp

from utils import (
    draw_violinplot,
    get_pssd,
    get_voxel_to_fsaverage_mapper,
    load_model_results, 
    participant_names,
)

model_type = "fasttext"
plt.switch_backend("agg")
figures_dir = f"../results/figures/consistency_across_participants_diff_pc_proj_{model_type}"
os.makedirs(figures_dir, exist_ok=True)

dpi = 200
threshold = 0.1
fontsize = 24

num_participants = len(participant_names)
###############################################################################
# Load Scores and Weights
###############################################################################
results_dict = load_model_results(model_type)
if model_type == "fasttext":
    do_flip_colors = True
elif model_type == "mBERT":
    do_flip_colors = False

all_primal_weights_dict = results_dict["all_primal_weights_dict"]
all_scores_dict = results_dict["all_scores_dict"]
weights_diff_scaled_dict = results_dict["weights_diff_scaled_dict"]

###############################################################################
# Load PCs
###############################################################################
pssd = get_pssd(model_type).squeeze()

###############################################################################
# Plot proj onto PCs
###############################################################################
cmap = "PRGn"
vmin = -0.5
vmax = 0.5

diff_proj_onto_diff_pc_group = dict()
if do_flip_colors:
    pssd = -pssd
for participant_idx, (participant_name, weights_diff) in enumerate(
    weights_diff_scaled_dict.items()
):
    voxel_to_fsaverage_mapper = get_voxel_to_fsaverage_mapper(participant_name)

    scores_en = all_scores_dict["en"][participant_name]
    scores_zh = all_scores_dict["zh"][participant_name]
    included_voxels = np.where((scores_en > threshold) & (scores_zh > threshold))
    included_voxels_mask = np.zeros(scores_en.size) * np.nan
    included_voxels_mask[included_voxels] = 1

    proj_diff = (
        pssd 
        @ (weights_diff / np.linalg.norm(weights_diff, axis=0))
        * included_voxels_mask
    )
    if participant_idx == 0:
        diff_proj_onto_diff_pc_group = (
            voxel_to_fsaverage_mapper @ proj_diff
        ).reshape(1, -1)
    else:
        diff_proj_onto_diff_pc_group = np.concatenate(
            [
                diff_proj_onto_diff_pc_group,
                (voxel_to_fsaverage_mapper @ proj_diff).reshape(1, -1),
            ],
            axis=0,
        )

offset = 0.1
fig, ax = plt.subplots(figsize=(9, 3))
for participant_idx in range(num_participants):
    other_participant_indices = np.setdiff1d(np.arange(num_participants), participant_idx)
    group_projection = np.nanmean(diff_proj_onto_diff_pc_group[other_participant_indices], axis=0)
    pos_vertices = np.where((group_projection > 0) & (~np.isnan(diff_proj_onto_diff_pc_group[participant_idx])))
    neg_vertices = np.where((group_projection < 0) & (~np.isnan(diff_proj_onto_diff_pc_group[participant_idx])))
    # t-test, z-transformed neg values < 0
    neg_values = diff_proj_onto_diff_pc_group[participant_idx][neg_vertices]
    pos_values = diff_proj_onto_diff_pc_group[participant_idx][pos_vertices]
    neg_pval = ttest_1samp(np.arctanh(neg_values), popmean=0, alternative="less").pvalue
    pos_pval = ttest_1samp(np.arctanh(pos_values), popmean=0, alternative="greater").pvalue
    print(participant_idx, "neg_pval", neg_pval, "pos_pval", pos_pval)
    print(np.mean(np.arctanh(neg_values)), np.mean(np.arctanh(pos_values)))
    print(np.mean(neg_values), np.mean(pos_values))
    plt.errorbar(participant_idx * 3 / 4 - offset, np.mean(neg_values), yerr=sem(neg_values), color="purple", capsize=5)
    draw_violinplot(neg_values, positions=[participant_idx * 3 / 4 - offset], color="purple")
    plt.errorbar(participant_idx * 3 / 4 + offset, np.mean(pos_values), yerr=sem(pos_values), color="green", capsize=5)
    draw_violinplot(pos_values, positions=[participant_idx * 3 / 4 + offset], color="green")
    plt.axhline(0, color="grey", linestyle="--", alpha=0.5)
legend_elements = [Patch(facecolor='green', label='Partial-Group PTSI > 0', alpha=0.3, edgecolor="green"),
                    Patch(facecolor='purple', label='Partial-Group PTSI < 0', alpha=0.3, edgecolor="purple")]
legend = plt.legend(handles=legend_elements, loc='lower center', fontsize=fontsize * 3 / 4, bbox_to_anchor=(0.5, 1), ncols=2)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(2)
ax.set_yticks([-0.5, 0, 0.5])
ax.set_yticklabels([-0.5, 0, 0.5], fontsize=fontsize)
plt.xticks(np.arange(num_participants) * 3 / 4, [f"P{i + 1}" for i in range(num_participants)], fontsize=fontsize)
plt.ylabel("PTSI", fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"{figures_dir}/proj_onto_pssd.png", dpi=dpi, bbox_extra_artists=[legend])
plt.close()

