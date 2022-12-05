#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# This script run the training of several models for Ensembling using ONLY CNC training data
set -euo pipefail

# you can use SunGrid Engine with this command:
# cmd="utils/queue.pl -l h='*'-V"
cmd_input="none"

# define several models where to do the evaluation
hf_models="bert-base-cased bert-base-uncased roberta-base roberta-large distilroberta-base facebook/bart-base microsoft/deberta-v3-base"

TRAIN_DATA=../data/train_subtask1.csv
DEV_DATA=../data/dev_subtask1.csv
TEST_DATA=../data/dev_subtask1.csv

# run per model 
echo "train and testing by folds"
for model_now in $(echo $hf_models); do
  (
    # getting the model and setting the output folder
    model=$model_now
    EXP_DIR=output/baselines_only_cnc/$(basename $model)

    output_dir=$EXP_DIR
    echo "Training with model: $model_now"
    
    # configure a GPU to use if we a defined 'CMD'
    if [ ! "$cmd_input" == "none" ]; then
        cmd="$cmd_input -N train_$(basename ${model})_baseline ${output_dir}/log/train_log.txt"
    else
        cmd=''
    fi

    $cmd bash subtask1/baselines/train_baselines.sh \
        --model $model \
        --save-steps "50000" \
        --eval-steps "100" \
        --logging-steps "100" \
        --train-epochs "50" \
        --output-dir $output_dir \
        --train-csv "$TRAIN_DATA" \
        --dev-csv "$DEV_DATA" \
        --test-csv "$TEST_DATA"

    # generate the outputs in the CNC challenge format
    # files to generate outputs:
    input_files="data/test_subtask1_text.csv data/dev_subtask1_text.csv"
    test_names="test_subtask1 dev_subtask1"

    # configure a GPU to use if we a defined 'CMD'
    if [ ! "$cmd_input" == "none" ]; then
        cmd="$cmd_input -N eval_$(basename ${model})_baseline ${output_dir}/log/eval_log.txt"
    else
        cmd=''
    fi

    $cmd bash subtask1/evaluations/eval_model.sh \
        --batch-size 24 \
        --input-model "$output_dir" \
        --input-files "$input_files" \
        --test-names "$test_names" 

    echo done running training for model: $model_now
    ) || touch $EXP_DIR/logs.error &
done
wait

if [ -f $EXP_DIR/logs.error ]; then
  echo "$0: something went wrong while fine-tuning the model"
  exit 1
fi


echo "Done training all models and evaluating them"

