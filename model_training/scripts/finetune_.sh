#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v3_on_flowers.sh
set -e

echo "Define your workspace"
read WORKSPACE
#WORKSPACE=/home/fabiohk/project

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${WORKSPACE}/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
echo "Folder name that training checkpoint and logs will be saved to"
read train_dir
TRAIN_DIR=${WORKSPACE}/${train_dir}
mkdir TRAIN_DIR -p

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi

# Select the model
echo "Choose the model (inception_v3 or mobilenet_v1)"
read model

if [ "$model" == "inception_v3" ]; then
  checkpoint_date="2016_08_28"
  model_="InceptionV3"
elif [ "$model" == "mobilenet_v1" ]; then
  model_download="mobilenet_v1_1.0_224"
  checkpoint_date="2017_06_14"
  exclude="MobilenetV1/AuxLogits,MobilenetV1/Logits"
else
  echo "Selected model does not exist or is not supported"
  echo "Finishing script..."
  exit
fi

if [ ! -d ${PRETRAINED_CHECKPOINT_DIR}/${model}/ ]; then
  mkdir -p ${PRETRAINED_CHECKPOINT_DIR}/${model}
  wget http://download.tensorflow.org/models/${model_download}_${checkpoint_date}.tar.gz
  tar -xvf ${model_download}_${checkpoint_date}.tar.gz -C ${PRETRAINED_CHECKPOINT_DIR}/${model}
  rm -v ${model_download}_${checkpoint_date}.tar.gz
fi

DATASET_DIR=${WORKSPACE}/train_data

# Ask if the user want to evaluate, train a new model, or resume training
echo "Want to evaluate or train? (0 to evaluate, 1 to train a new model, or 2 to resume training"
read option

if [ "$option" == 0 ]; then
  python eval_image_classifier2.py \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${TRAIN_DIR} \
    --dataset_name=ervas \
    --dataset_split_name=validation \
    --batch_size=32 \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model}
elif [ "$option" == 1 ]; then
  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=ervas \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model} \
    --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${model} \
    --checkpoint_exclude_scopes=${exclude} \
#    --trainable_scopes=${exclude} \
    --max_number_of_steps=128238 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --end_learning_rate=0.00001\
    --save_interval_secs=3600 \
    --save_summaries_secs=3600 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004

  python eval_image_classifier2.py \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${TRAIN_DIR} \
    --dataset_name=ervas \
    --dataset_split_name=validation \
    --batch_size=32 \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model}

  echo "Exponential decay: 0.001 to 0.00001"

  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/exp_decay \
    --dataset_name=ervas \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model} \
    --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${model} \
    --checkpoint_exclude_scopes=${exclude} \
    --max_number_of_steps=50000 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --end_learning_rate=0.00001\
    --save_interval_secs=600 \
    --save_summaries_secs=600 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004

  python eval_image_classifier2.py \
    --checkpoint_path=${TRAIN_DIR}/exp_decay \
    --eval_dir=${TRAIN_DIR} \
    --dataset_name=ervas \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model}

  echo "Fixed: 0.001"

  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/fixed_0.001 \
    --dataset_name=ervas \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model} \
    --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${model} \
    --checkpoint_exclude_scopes=${exclude} \
    --max_number_of_steps=50000 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=600 \
    --save_summaries_secs=600 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004

  python eval_image_classifier2.py \
    --checkpoint_path=${TRAIN_DIR}/fixed_0.001 \
    --eval_dir=${TRAIN_DIR} \
    --dataset_name=ervas \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model}

  echo "Fixed: 0.0001"

  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/fixed_0.0001 \
    --dataset_name=ervas \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model} \
    --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${model} \
    --checkpoint_exclude_scopes=${exclude} \
    --max_number_of_steps=50000 \
    --batch_size=32 \
    --learning_rate=0.0001 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=600 \
    --save_summaries_secs=600 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004

  python eval_image_classifier2.py \
    --checkpoint_path=${TRAIN_DIR}/fixed_0.0001 \
    --eval_dir=${TRAIN_DIR} \
    --dataset_name=ervas \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model}

  echo "Fixed: 0.00001"

  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/fixed_0.00001 \
    --dataset_name=ervas \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model} \
    --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${model} \
    --checkpoint_exclude_scopes=${exclude} \
    --max_number_of_steps=50000 \
    --batch_size=32 \
    --learning_rate=0.00001 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=600 \
    --save_summaries_secs=600 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004

  python eval_image_classifier2.py \
    --checkpoint_path=${TRAIN_DIR}/fixed_0.00001 \
    --eval_dir=${TRAIN_DIR} \
    --dataset_name=ervas \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${model}

else
   echo "Nothing..."
fi
