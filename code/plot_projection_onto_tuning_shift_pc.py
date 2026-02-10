"""Plot projection onto the primary tuning shift dimension.

Figure 4B.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from utils import (
    do_flip_colors_dict,
    get_pssd,
    get_voxel_to_fsaverage_mapper,
    load_model_results,
    num_fsaverage_vertices,
    participant_names,
    save_flatmap_image,
    scores_threshold,
    template_surface,
)

model_type = "fasttext"
plt.switch_backend("agg")
figures_dir = f"../results/figures/diff_pc_proj_{model_type}"
os.makedirs(figures_dir, exist_ok=True)

dpi = 200
do_plot_group = True
cmap = "PRGn"
vmin = -0.5
vmax = 0.5
num_participants = len(participant_names)
min_participants_for_group = 1
###############################################################################
# Load Scores and Weights
###############################################################################
results_dict = load_model_results(model_type)

all_scores_dict = results_dict["all_scores_dict"]
weights_diff_mean_lang_dict = results_dict["weights_diff_scaled_dict"]

###############################################################################
# Load PCs
###############################################################################
pssd = get_pssd(model_type)
if do_flip_colors_dict[model_type]:
    pssd *= -1

###############################################################################
# Plot proj onto PCs
###############################################################################
diff_proj_onto_diff_pc_group = None

for participant_idx, (participant_name, weights_diff) in enumerate(
    weights_diff_mean_lang_dict.items()
):
    voxel_to_fsaverage_mapper = get_voxel_to_fsaverage_mapper(participant_name)
    # Get voxels to plot.
    scores_en = all_scores_dict["en"][participant_name]
    scores_zh = all_scores_dict["zh"][participant_name]
    included_voxels = np.where((scores_en > scores_threshold) & (scores_zh > scores_threshold))
    included_voxels_mask = np.zeros(scores_en.size) * np.nan
    included_voxels_mask[included_voxels] = 1
    included_voxels_mask_fsaverage = voxel_to_fsaverage_mapper @ np.nan_to_num(included_voxels_mask)
    included_voxels_mask_fsaverage[included_voxels_mask_fsaverage == 0] = np.nan
    included_voxels_mask_fsaverage[included_voxels_mask_fsaverage > 0] = 1

    # Projection onto tuning shift dimension.
    proj_diff = (
       pssd 
        @ (weights_diff / np.linalg.norm(weights_diff, axis=0))
    ).squeeze()
    if diff_proj_onto_diff_pc_group is None:
        diff_proj_onto_diff_pc_group = (
            voxel_to_fsaverage_mapper @ proj_diff * included_voxels_mask_fsaverage
        ).reshape(1, -1)
    else:
        diff_proj_onto_diff_pc_group = np.concatenate(
            [
                diff_proj_onto_diff_pc_group,
                (voxel_to_fsaverage_mapper @ proj_diff * included_voxels_mask_fsaverage).reshape(1, -1),
            ],
            axis=0,
        )
    proj_diff *= included_voxels_mask
    save_flatmap_image(proj_diff, participant_name, save_path=os.path.join(figures_dir, f"{participant_name}_{model_type}"), cmap=cmap, vmin=vmin, vmax=vmax) 

# Plot for group
not_nan_mask = np.ones(num_fsaverage_vertices) * np.nan
not_nan_mask[
    np.where(
        (~np.isnan(diff_proj_onto_diff_pc_group)).sum(0)
        >= min_participants_for_group
    )
] = 1
save_flatmap_image(np.nanmean(diff_proj_onto_diff_pc_group, 0) * not_nan_mask,template_surface, save_path=os.path.join(figures_dir, f"group_{model_type}"), cmap=cmap, vmin=vmin, vmax=vmax) 
