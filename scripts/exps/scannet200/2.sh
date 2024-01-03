#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=100
CURR_QUERY=100

# # TRAIN
python main_instance_segmentation.py \
general.experiment_name="2_scannet200_signed_random" \
general.project_name="New_Project" \
data/datasets=scannet200 \
general.num_targets=201 \
data.num_labels=200 \
general.eval_on_segments=true \
general.train_on_segments=true \
general.perturbation_type="signed_random" \
general.use_non_perturbed=false \
general.use_perturbed=true \
general.sim_loss_weight=0.5 \
general.perturbation_magnitude=0.001

# # TEST
# python main_instance_segmentation.py \
# general.experiment_name="scannet200_val_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
# general.project_name="scannet200_eval" \
# general.checkpoint="checkpoints/scannet200/scannet200_val.ckpt" \
# data/datasets=scannet200 \
# general.num_targets=201 \
# data.num_labels=200 \
# general.eval_on_segments=true \
# general.train_on_segments=true \
# general.train_mode=false \
# model.num_queries=${CURR_QUERY} \
# general.topk_per_image=${CURR_TOPK} \
# general.use_dbscan=false \
# general.dbscan_eps=${CURR_DBSCAN}
