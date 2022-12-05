#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# script to evaluate a file, or several files!
set -euo pipefail

batch_size=24
input_model= ./output/kfold/bert-base/
input_files="../data/test_subtask1_text.csv ../data/dev_subtask1_text.csv"
test_names="test_subtask1 dev_subtask1"
run_ensemble=false

# with this we can parse options to this script
. ./utils/parse_options.sh

# check if you want to get the logits instead of prepare to CNC
if [ "$run_ensemble" = "true" ]; then
  optional_args="--make-ensemble-outs"
else
  optional_args=""
fi

# running the command
python3 evaluations/run_inference.py \
  --batch-size $batch_size \
  --input-model "$input_model" \
  --input-files "$input_files" \
  --test-names "$test_names" \
  --output-folder $input_model/evaluations \
  $optional_args

echo Done
exit 0
