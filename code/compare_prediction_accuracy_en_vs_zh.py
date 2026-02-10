"""Bar plot and significance test of en vs zh prediction accuracy.

Figure 2B.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import (
    color_en,
    color_zh,
    get_voxels_for_region,
    load_model_results,
    participant_names,
    participant_to_marker_dict
)

model_type = "fasttext"
figures_dir = f"../results/figures/prediction_accuracy_en_vs_zh_{model_type}"
os.makedirs(figures_dir, exist_ok=True)
results_dict = load_model_results(model_type)
do_permutation_test = False

offset = 0.2
width = 0.35
s = 70
fontsize = 16
pvalue_threshold = 0.05
os.makedirs(figures_dir, exist_ok=True)
hemis = ["lh", "rh"]
region_names_to_plot = ["temporal", "parietal", "prefrontal"]

def permutation_test(accuracy_en, accuracy_zh, n_iter: int = 1000):
    true_diff = np.mean(accuracy_zh) - np.mean(accuracy_en)
    num_perm_gt_true = 0
    all_vals = np.concatenate([accuracy_en, accuracy_zh])
    for _ in range(n_iter):
        indices_en = np.random.choice(len(all_vals), len(accuracy_en), replace=False)
        indices_zh = np.array([idx for idx in range(len(all_vals)) if idx not in indices_en])
        perm_diff = np.mean(all_vals[indices_zh]) - np.mean(all_vals[indices_en])
        if perm_diff > true_diff:
            num_perm_gt_true += 1
    return num_perm_gt_true / n_iter

average_accuracy_per_region = defaultdict(lambda: {"en": [], "zh": []})
for participant_name in participant_names:
    accuracy_en = np.nan_to_num(results_dict["all_scores_dict"]["en"][participant_name])
    accuracy_zh = np.nan_to_num(results_dict["all_scores_dict"]["zh"][participant_name])


    r2_en = accuracy_en ** 2  # Square because saved as sqrt R2.
    r2_zh = accuracy_zh ** 2  # Square because saved as sqrt R2.
    region_to_voxels_dict = get_voxels_for_region(participant_name)
    print(participant_name)
    for hemi_name in hemis:
        for region_name in region_names_to_plot:
            region_voxels = region_to_voxels_dict[region_name + hemi_name]
            if do_permutation_test:
                pvalue = permutation_test(r2_en[region_voxels], r2_zh[region_voxels])
                if pvalue > pvalue_threshold:
                    print("*******", participant_name, region_name + hemi_name, pvalue)
            average_accuracy_per_region[region_name + hemi_name]["en"].append(np.nanmean(accuracy_en[region_voxels]))
            average_accuracy_per_region[region_name + hemi_name]["zh"].append(np.nanmean(accuracy_zh[region_voxels]))

all_region_names_to_plot = []
for hemi_name in hemis:
    all_region_names_to_plot += [region_name + hemi_name for region_name in region_names_to_plot]

fig, ax = plt.subplots(figsize=(16, 5))
for region_idx, region_name in enumerate(all_region_names_to_plot):
    plt.bar(region_idx - offset, np.mean(average_accuracy_per_region[region_name]["en"]), color=color_en, alpha=0.5, width=width)
    plt.bar(region_idx + offset, np.mean(average_accuracy_per_region[region_name]["zh"]), color=color_zh, alpha=0.5, width=width)
    for participant_idx, participant_name in enumerate(participant_names):
        plt.scatter(region_idx - offset, average_accuracy_per_region[region_name]["en"][participant_idx], color=color_en, marker=participant_to_marker_dict[participant_name], s=s, edgecolors="white")
        plt.scatter(region_idx + offset, average_accuracy_per_region[region_name]["zh"][participant_idx], color=color_zh, marker=participant_to_marker_dict[participant_name], s=s, edgecolors="white")
plt.style.use('default')
plt.xticks(range(len(all_region_names_to_plot)), [region_name.capitalize().replace("lh", "\nLH").replace("rh", "\nRH") for region_name in all_region_names_to_plot], fontsize=fontsize)
plt.ylabel(r"Average $\sqrt{R^2}$" + "\nover Voxels", fontsize=fontsize)
ax.set_yticks([0, 0.05])
ax.set_yticklabels([0, 0.05], fontsize=fontsize)
plt.ylim([0, 0.05])
legend_elements = [Patch(facecolor=color_en, edgecolor=color_en, label='English'),
                   Patch(facecolor=color_zh, edgecolor=color_zh, label='Chinese')] + [Line2D([0], [0], marker=participant_to_marker_dict[participant_name], markerfacecolor='k', markersize=10, lw=0, label=f"P{participant_names.index(participant_name) + 1}", markeredgecolor='white', alpha=0.5)
                for participant_name in participant_names]
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(2)
plt.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=fontsize, bbox_to_anchor=(0.5, 1))
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, f"average_prediction_accuracy_per_region_{model_type}.png"), dpi=100)
plt.close()
