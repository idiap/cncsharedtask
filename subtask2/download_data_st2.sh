#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License 

mkdir data && cd data || exit 1

echo "Downloading dataset..."
DATA_URL="https://raw.githubusercontent.com/tanfiona/CausalNewsCorpus/master/data/" || exit 1
wget "${DATA_URL}train_subtask2.csv" || exit 1
wget "${DATA_URL}dev_subtask2.csv" || exit 1
wget "${DATA_URL}dev_subtask2_grouped.csv" || exit 1
wget "${DATA_URL}test_subtask2_text.csv" || exit 1

echo "Downloading checkpoints"
######https://www.fit.vutbr.cz/~ifajcik/pubdata/
wget "https://www.fit.vutbr.cz/~ifajcik/pubdata/case22/checkpoint_t5ces_large.zip" || exit 1
unzip "checkpoint_t5ces_large.zip" && rm "checkpoint_t5ces_large.zip" || exit 1
wget "https://www.fit.vutbr.cz/~ifajcik/pubdata/case22/checkpoint_t5ces_base.zip" || exit 1
unzip "checkpoint_t5ces_base.zip" && rm "checkpoint_t5ces_base.zip" || exit 1
wget "https://www.fit.vutbr.cz/~ifajcik/pubdata/case22/checkpoint_t5ecs_base.zip" || exit 1
unzip "checkpoint_t5ecs_base.zip" && rm "checkpoint_t5ecs_base.zip" || exit 1
wget "https://www.fit.vutbr.cz/~ifajcik/pubdata/case22/checkpoint_t5ces_nohistory.zip" || exit 1
unzip "checkpoint_t5ces_nohistory.zip" && rm "checkpoint_t5ces_nohistory.zip" || exit 1
cd .. || exit 1
echo "Subtask2 data downloaded successfully!"