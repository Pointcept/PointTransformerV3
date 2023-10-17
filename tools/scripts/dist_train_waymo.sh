#!/usr/bin/env bash

PYTHON=python

DATASET=waymo
CONFIG=centerpoint
TAG=default
GPU=None
WORKERS=2

while getopts "p:c:t:g:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    t)
      TAG=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo "DDP Port: $PORT"

CFG_NAME=${DATASET}_models/${CONFIG}

echo " =========> RUN TASK <========="
$PYTHON -m torch.distributed.launch --nproc_per_node=${GPU} --rdzv_endpoint=localhost:${PORT} \
train.py \
--launcher pytorch \
--cfg_file cfgs/${CFG_NAME}.yaml \
--workers ${WORKERS} \
--extra_tag $TAG \
--max_ckpt_save_num 1 \
--num_epochs_to_eval 1 \

#echo " =========> EVAL TASK <========="
#EPOCH=epoch_30
#WAYMO_GT=../data/waymo/gt.bin
#WAYMO_EVAL=../data/waymo/compute_detection_metrics_main
#WAYMO_EXP_DIR=../output/$CFG_NAME/$TAG/eval/eval_with_train/$EPOCH/val/final_result/data
#
#$WAYMO_EVAL ${WAYMO_EXP_DIR}/detection_pred.bin ${WAYMO_GT} | tee ${WAYMO_EXP_DIR}/metric.txt
