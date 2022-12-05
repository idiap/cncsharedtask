#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# You need to run this bash script from the main directory! subtask1/
K=5
SEED=42

databases="train_subtask1.csv"

for db in $(echo $databases); do
  # gen kfolds for $db 
  INPUT_CSV=data/$db
  SAVE_DIR=data/folds_subtask1/$(basename $INPUT_CSV .csv)

  echo "generating folds for: $(basename $INPUT_CSV) in: $SAVE_DIR"
  python3 data/scripts/gen_kfolds.py \
    --input_csv $INPUT_CSV --k $K --seed $SEED --save_dir $SAVE_DIR
done

echo " done creating folds training data. Nb. Folds: $K"
exit 0
