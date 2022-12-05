#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score

DECISION_TRESHOLD = 0.5

NUM_K = sys.argv[1]
SEED = sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else None
CMD_IX = sys.argv[4] if len(sys.argv) > 4 else None

# NUM_SAMPLES = [5, 6, 3, 9]#, 7, 8, 4]
NUM_SAMPLES = [4, 5]

PATH_RESTULTS = "result/"
# PATH_MODELS = []
# for num_sample in NUM_SAMPLES:
#     PATH_MODELS.append(PATH_RESTULTS + f"tmp-{NUM_K}-{SEED}-{MODEL}-{CMD_IX}-num_sample-{num_sample}")

PATH_MODELS = [
    PATH_RESTULTS + "tmp-356-21-roberta-large-6-dfr0.50-num_sample-3",
    PATH_RESTULTS + "tmp-356-21-roberta-large-6-dfr0.50-num_sample-2",
    PATH_RESTULTS + "tmp-356-21-roberta-large-12-dfr0.50",
]

for dname, dcsv in [("eval", "dev"), ("dev", "test"), ("test", "test_competition")]:
    print("\n" + ("=" * 180))
    print(dname.upper())

    df = pd.read_csv(f"data/k-shot/CNC/{NUM_K}-{SEED}/{dcsv}.csv", header=None)
    index = df.iloc[:, -1].values.tolist()
    y_true = df.iloc[:, 0].values

    # Logits
    y_pred_all = []
    mean_logits = None
    for m_ix, path in enumerate(PATH_MODELS):
        logits = np.load(os.path.join(path, f"{dname}_cnc.npy"))

        y_prob = softmax(logits, axis=1)
        y_pred = [1 if p[1] > DECISION_TRESHOLD else 0 for p in y_prob]
        y_pred_all.append(y_pred)

        if dname != "test":
            print(classification_report(y_true, y_pred, zero_division=0))
            print()
            print(
                f"f1_score: {f1_score(y_true, y_pred, zero_division=0)} ({PATH_MODELS[m_ix]})"
            )
        if mean_logits is None:
            mean_logits = logits
        else:
            mean_logits += logits
    mean_logits /= len(PATH_MODELS)

    y_prob = softmax(mean_logits, axis=1)
    y_pred = [1 if p[1] > DECISION_TRESHOLD else 0 for p in y_prob]

    if dname != "test":
        print(classification_report(y_true, y_pred, zero_division=0))
        print()
        print("\nAVG f1_score", f1_score(y_true, y_pred, zero_division=0))

    with open(f"{dname}_cnc_avg.json", "w") as writer:
        for e_ix, index in enumerate(index):
            writer.write(
                f'{{"index": "{index}", "prediction": {y_pred[e_ix]}, "truth": {y_true[e_ix]}}}\n'
            )

    y_pred = []
    for y_ix in range(len(y_true)):
        pos_votes = [
            1 for m_ix in range(len(PATH_MODELS)) if y_pred_all[m_ix][y_ix] == 1
        ]
        neg_votes = [
            0 for m_ix in range(len(PATH_MODELS)) if y_pred_all[m_ix][y_ix] == 0
        ]
        y_pred.append(1 if len(pos_votes) > len(neg_votes) else 0)

    if dname != "test":
        # print(classification_report(y_true, y_pred, zero_division=0))
        # print()
        print("VOT f1_score", f1_score(y_true, y_pred, zero_division=0))

    with open(f"{dname}_cnc_vot.json", "w") as writer:
        for e_ix, index in enumerate(index):
            writer.write(
                f'{{"index": "{index}", "prediction": {y_pred[e_ix]}, "truth": {y_true[e_ix]}}}\n'
            )
