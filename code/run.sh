#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.

echo "Plotting prediction accuracy for each participant (Figure 2A)."
python plot_prediction_accuracy.py

echo "Creating bar plots and performing permutation test for en vs zh accuracy (Figure 2B)."
python compare_prediction_accuracy_en_vs_zh.py

echo "Saving interpretation of cluster centroids (Figure 3A) and of PSSD (Figure 4A)."
python interpret_pcs.py

echo "Plotting semantic cluster flatmaps (Figure 3B), confusion matrix between cluster assignments (Figure 3C), and PTSI per cluster (Figure 5A)."
python plot_semantic_clusters.py

echo "Plotting projections onto PSSD (Figure 4B)."
python plot_projection_onto_tuning_shift_pc.py

echo "Plotting consistency of vertexwise PTSI across participants."
python projection_onto_tuning_shift_pc_consistency_btwn_participants.py

echo "Saving closest words for example voxels (Figure 5B)."
python get_example_words.py
