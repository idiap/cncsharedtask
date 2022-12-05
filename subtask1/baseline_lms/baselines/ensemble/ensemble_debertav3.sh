#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# This script run the training several models for Ensembling using ONLY CNC training data
set -euo pipefail

# you can use SunGrid Engine with this command:
# cmd="utils/queue.pl -l h='*'-V"
cmd_input="none"

# define several models where to do the evaluation
hf_models="bert-base-cased roberta-base facebook/bart-base microsoft/deberta-v3-base"

TRAIN_DATA=../data/train_subtask1.csv
DEV_DATA=../data/dev_subtask1.csv
TEST_DATA=../data/dev_subtask1.csv

output_folder=output/ensemble
# producing the label files neeeded by the script
mkdir -p $output_folder/results

cat $DEV_DATA | rev | cut -d',' -f4 | rev | tail +2 > $output_folder/results/labels.txt
cat $DEV_DATA tail +2 | cut -d',' -f1 > $output_folder/results/dev_ids.txt

# write the ids for the test set
cat $TEST_DATA tail +2 | cut -d',' -f1 > $output_folder/results/test_ids.txt

nb_models=50

# run $nb_models models with different seeds 
echo "train $nb_models models with different seeds for ensemble experiments"

for model_now in $(echo $hf_models); do

  echo "training $nb_models $model_now for model ensembling"
  for seed in $(seq 1 1 $nb_models); do
    (

      # sleep to not submit all at once:
      sleep $(echo $(( $seed * 10 )))
      # getting the model and setting the output folder
      model=$model_now
      EXP_DIR=$output_folder/$seed/$(basename $model)

      output_dir=$EXP_DIR
      echo "Training with model: $model, seed: $seed"
      
      # configure a GPU to use if we a defined 'CMD'
      if [ ! "$cmd_input" == "none" ]; then
        cmd="$cmd_input -N ensemble_train_$(basename ${model}) ${output_dir}/log/train_log.txt"
      else
        cmd=''
      fi

      $cmd bash baselines/train_baselines.sh \
          --model "$model" \
          --seed "$seed" \
          --save-steps "50000" \
          --eval-steps "100" \
          --logging-steps "100" \
          --train-epochs "50" \
          --output-dir "$output_dir" \
          --train-csv "$TRAIN_DATA" \
          --dev-csv "$DEV_DATA" \
          --test-csv "$TEST_DATA"

      # generate the outputs in the CNC challenge format
      # files to generate outputs:
      input_files="data/test_subtask1_text.csv data/dev_subtask1_text.csv"
      test_names="test_subtask1 dev_subtask1"

      $cmd bash evaluations/eval_model.sh \
          --run-ensemble "true" \
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

done

# configure a GPU to use if we a defined 'CMD'
if [ ! "$cmd_input" == "none" ]; then
    cmd="$cmd_input -N merge_ensemble $output_folder/ensemble_log.txt"
else
    cmd=''
fi

# running the ensemble script:
$cmd python3 baselines/ensemble/get_ensemble_f1.py \
  --num-runs "10000" \
  --models "*" \
  --ensemble-folder "$output_folder/" \
  --out-folder "$output_folder/results/" \
  --labels-file "$output_folder/results/labels.txt"

echo "Done training, evaluation and ensemble of models"

