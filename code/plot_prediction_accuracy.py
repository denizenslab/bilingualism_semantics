"""Plot voxelwise prediction accuracy for each participant.

Figure 2A.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from collections import defaultdict

from utils import (
    get_voxel_to_fsaverage_mapper,
    languages,
    load_model_results,
    num_fsaverage_vertices,
    participant_names,
    save_flatmap_image,
    template_surface
)

model_type = "fasttext"
figures_dir = f"../results/figures/prediction_accuracy_{model_type}"
os.makedirs(figures_dir, exist_ok=True)
results_dict = load_model_results(model_type)
plt.switch_backend("agg")

vmin = 0
vmax = 0.2
vmax_group = 0.1
cmap = "inferno"

scores_dict_across_language_group = defaultdict(
    lambda: np.zeros(num_fsaverage_vertices)
)
scores_dict_within_language_group = defaultdict(
    lambda: np.zeros(num_fsaverage_vertices)
)

for participant_name in participant_names:
    voxel_to_fsaverage_mapper = get_voxel_to_fsaverage_mapper(participant_name)

    # Plot within-language performance.
    scores_dict_within_language = {}
    for language in languages:
        scores = results_dict["all_scores_dict"][language][participant_name]
        scores_dict_within_language[language] = scores

        # Group-level
        scores_dict_within_language_group[language] += (
            scores @ voxel_to_fsaverage_mapper.T
        )

        save_flatmap_image(scores, participant_name, save_path=os.path.join(figures_dir, f"{participant_name}_{language}_{model_type}"), cmap=cmap, vmin=vmin, vmax=vmax) 

    # Plot cross-language performance.
    scores_dict_across_language = {}
    for train_language, test_language in zip(languages, languages[::-1]):
        scores = results_dict["all_across_scores_dict"][train_language][
            participant_name
        ]
        scores_dict_across_language[(train_language, test_language)] = scores
        scores_dict_across_language_group[(train_language, test_language)] += (
            scores @ voxel_to_fsaverage_mapper.T
        )

        save_flatmap_image(scores, participant_name, save_path=os.path.join(figures_dir, f"{participant_name}_crosspred_train_{train_language}_test_{test_language}_{model_type}"), cmap=cmap, vmin=vmin, vmax=vmax) 


# Group level
for k, v in scores_dict_across_language_group.items():
    scores_dict_across_language_group[k] = v / len(participant_names)
for k, v in scores_dict_within_language_group.items():
    scores_dict_within_language_group[k] = v / len(participant_names)

# Within-language
for language in languages:
    save_flatmap_image(scores_dict_within_language_group[language], template_surface, save_path=os.path.join(figures_dir, f"group_{language}_{model_type}"), cmap=cmap, vmin=vmin, vmax=vmax_group) 

# Across-language
for train_language, test_language in zip(languages, languages[::-1]):
    save_flatmap_image(scores_dict_across_language_group[(train_language, test_language)], template_surface, save_path=os.path.join(figures_dir, f"group_crosspred_train_{train_language}_test_{test_language}_{model_type}"), cmap=cmap, vmin=vmin, vmax=vmax_group) 
