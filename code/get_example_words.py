"""Get words for example voxels.

Figure 5B.
"""
import numpy as np
import os
import pickle

from utils import (
    get_clusters_dicts,
    get_stimulus_word_embeddings,
    get_vector_top_bottom_word_projections,
    load_model_results,
)

model_type = "fasttext"
participant_name = "p01"
voxels_to_plot = [43064, 10490]
num_words_to_plot = 10

clusters_dict_en, clusters_dict_zh = get_clusters_dicts(model_type)
results_dict = load_model_results(model_type)

weights_en = results_dict["all_primal_weights_dict"]["en"][participant_name]
weights_zh = results_dict["all_primal_weights_dict"]["zh"][participant_name]

weights_en /= np.linalg.norm(weights_en, axis=0).reshape(1, -1)
weights_zh /= np.linalg.norm(weights_zh, axis=0).reshape(1, -1)

(word_embedding_matrix_en,
word_embedding_matrix_zh,
words_list_en,
words_list_zh) = get_stimulus_word_embeddings(model_type)
clusters_dict_en, _ = get_clusters_dicts(model_type.replace("_sig_mask", ""))
cluster_centroids = clusters_dict_en["cluster_avgs"]
words_list = np.array(words_list_en)

for v in voxels_to_plot:
    words = get_vector_top_bottom_word_projections(
        vectors=np.concatenate([weights_en[:, v].reshape(1, -1), weights_zh[:, v].reshape(1, -1)], axis=0),
        words_list=np.array(words_list_en),
        word_embedding_matrix=word_embedding_matrix_en,
    )
    cluster_en = clusters_dict_en[participant_name][v]
    cluster_zh = clusters_dict_zh[participant_name][v]
    print(cluster_en, cluster_zh)

    words_en = words[0]["top"][:num_words_to_plot]
    words_zh = words[1]["top"][:num_words_to_plot]
    with open(f"../results/outputs/example_words_{v}.p", "wb") as f:
        pickle.dump({"cluster_en": cluster_en,
            "cluster_zh": cluster_zh,
            "words_en": words_en,
            "words_zh": words_zh}, f)
