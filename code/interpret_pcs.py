"""Get interpretation of PCs and cluster centroids.

Figures 3A, 4A.
"""
import numpy as np
import pickle

from utils import get_closest_farthest_words, get_clusters_dicts, get_vector_top_bottom_word_projections, get_stimulus_word_embeddings, get_pssd


if __name__ == "__main__":
    model_type = "fasttext"
    pssd = get_pssd(model_type)

    (word_embedding_matrix_en,
    word_embedding_matrix_zh,
    words_list_en,
    words_list_zh) = get_stimulus_word_embeddings(model_type)
    clusters_dict_en, _ = get_clusters_dicts(model_type)
    cluster_centroids = clusters_dict_en["cluster_avgs"]
    if model_type in ["fasttext"]:
        # Get word projections.
        words_list = np.array(words_list_en)
        word_projections_dict = {}
        word_projections_dict["en"] = get_vector_top_bottom_word_projections(
            vectors=pssd.reshape(1, -1),
            words_list=np.array(words_list_en),
            word_embedding_matrix=word_embedding_matrix_en,
        )
        word_projections_dict["zh"] = get_vector_top_bottom_word_projections(
            vectors=pssd.reshape(1, -1),
            words_list=np.array(words_list_zh),
            word_embedding_matrix=word_embedding_matrix_zh,
        )
        word_projections_dict[
            "clusters_en_words"
        ] = get_vector_top_bottom_word_projections(
            vectors=cluster_centroids,
            words_list=np.array(words_list_en),
            word_embedding_matrix=word_embedding_matrix_en,
        )
        word_projections_dict[
            "clusters_zh_words"
        ] = get_vector_top_bottom_word_projections(
            vectors=cluster_centroids,
            words_list=np.array(words_list_zh),
            word_embedding_matrix=word_embedding_matrix_zh,
        )
    elif model_type == "mBERT":
        counts_per_word = [
            np.count_nonzero(words_list_en == word) for word in np.unique(words_list_en)
        ]
        threshold_en = np.nanpercentile(counts_per_word, 99)

        counts_per_word = [
            np.count_nonzero(words_list_zh == word) for word in np.unique(words_list_zh)
        ]
        threshold_zh = np.nanpercentile(counts_per_word, 99)

        # Get word projections.
        words_list = np.array(words_list_en)
        word_projections_dict = {}
        word_projections_dict["en"] = get_closest_farthest_words(
            vectors=pssd.reshape(1, -1),
            words_list=np.array(words_list_en),
            word_embedding_matrix=word_embedding_matrix_en,
            threshold=threshold_en,
        )
        word_projections_dict["zh"] = get_closest_farthest_words(
            vectors=pssd.reshape(1, -1),
            words_list=np.array(words_list_zh),
            word_embedding_matrix=word_embedding_matrix_zh,
            threshold=threshold_zh,
        )
        word_projections_dict["clusters_en_words"] = get_closest_farthest_words(
            vectors=cluster_centroids,
            words_list=np.array(words_list_en),
            word_embedding_matrix=word_embedding_matrix_en,
            threshold=threshold_en,
        )
        word_projections_dict["clusters_zh_words"] = get_closest_farthest_words(
            vectors=cluster_centroids,
            words_list=np.array(words_list_zh),
            word_embedding_matrix=word_embedding_matrix_zh,
            threshold=threshold_zh,
        )

    with open(f"../results/outputs/word_projections_{model_type}.p", "wb") as f:
        pickle.dump(word_projections_dict, f)
