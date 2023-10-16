#!/bin/sh

TASK=train
ENV=pcdet-torch1.10.1-cuda11.3-py3.8
PYTHON=python
DATASET=waymo
CONFIG=centerpoint
EXP_NAME=default

NUM_GPU=4
NUM_MACHINE=1
NUM_CPU_PER_GPU=12

WEIGHT="None"
RESUME=false

while getopts "t:e:p:d:c:n:g:m:u:w:r" opt; do
  case $opt in
    t)
      TASK=$OPTARG
      ;;
    e)
      ENV=$OPTARG
      ;;
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    u)
      NUM_CPU_PER_GPU=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

SCRIPTS_DIR=$(cd $(dirname "$0");pwd) || exit

NUM_CPU=`expr $NUM_CPU_PER_GPU \* $NUM_GPU`

tmux new-session -d -s "pcdet-${TASK}-${DATASET}-${CONFIG}-${EXP_NAME}"
tmux send-keys -t "pcdet-${TASK}-${DATASET}-${CONFIG}-${EXP_NAME}" "
  source /mnt/petrelfs/share_data/wuxiaoyang/scripts/activate_env/${ENV}.sh
  " Enter

if [ "${TASK}" == 'train' ]
then
  tmux send-keys -t "pcdet-${TASK}-${DATASET}-${CONFIG}-${EXP_NAME}" "
  srun --preempt -u -p Ai4sci_3D --job-name=pcdet-${TASK}-${DATASET}-${CONFIG}-${EXP_NAME} --gres=gpu:$NUM_GPU --nodes=$NUM_MACHINE --ntasks-per-node=1 --cpus-per-task=$NUM_CPU --kill-on-bad-exit \
  sh $SCRIPTS_DIR/dist_train_waymo.sh -p $PYTHON -g $NUM_GPU -c $CONFIG -t $EXP_NAME
  " Enter

fi
