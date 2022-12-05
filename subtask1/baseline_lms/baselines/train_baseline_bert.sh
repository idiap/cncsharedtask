#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# script to train the baselines for CNC challenge
set -euo pipefail

# you can use SunGrid Engine with this command:
# cmd="utils/queue.pl -l h='*'-V"
cmd_input="none"

# model related vars
model=bert-base-cased 
save_steps=50000
eval_steps=100
logging_steps=100
train_epochs=50
seed=1234

output_dir=./output

# folds are in data/folds/
train_csv=../data/train_subtask1.csv
dev_csv=../data/dev_subtask1.csv
test_csv=../data/dev_subtask1.csv

# with this we can parse options from CLI
. ./utils/parse_options.sh

train_basename=$(basename $train_csv .csv)

# run the training:
python3 run_case.py \
  --report_to "none" \
  --seed "$seed" \
  --run_name "${model}_epochs${train_epochs}_$train_basename" \
  --eval_steps=$eval_steps \
  --logging_steps=$logging_steps \
  --task_name cola \
  --train_file $train_csv --do_train \
  --validation_file $dev_csv --do_eval \
  --test_file $test_csv --do_predict \
  --num_train_epochs $train_epochs --save_steps $save_steps \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --model_name_or_path $model \
  --output_dir $output_dir --overwrite_output_dir

# create the file to submit to CNC challenge:
python3 reformat_output.py \
  $output_dir/predict_results_cola.txt \
  $output_dir/predict_results_cnc_challenge.json

echo done training model in $output_dir
