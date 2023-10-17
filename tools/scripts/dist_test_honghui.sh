#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0,1,2,3

NGPUS=4

CFG_NAME=waymo_models/centerpoint
TAG_NAME=default

EPOCH=epoch_30
CKPT=../output/$CFG_NAME/$TAG_NAME/ckpt/checkpoint_$EPOCH.pth

python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 4 --extra_tag $TAG_NAME --ckpt $CKPT

GT=../data/waymo/gt.bin
EVAL=../data/waymo/compute_detection_metrics_main
DT_DIR=../output/$CFG_NAME/$TAG_NAME/eval/$EPOCH/val/default/final_result/data

$EVAL $DT_DIR/detection_pred.bin $GT > $DT_DIR/metric.txt
