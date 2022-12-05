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
import sys

import numpy as np
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score

PATH_EXPERIMENT_RESULTS = sys.argv[1]


with open(os.path.join(PATH_EXPERIMENT_RESULTS, "dev_cnc.json"), "r") as reader:
    lines = reader.read().split("\n")
    results = [json.loads(line) for line in lines if line]


y_pred = [r["prediction"] for r in results]
y_true = [r["truth"] for r in results]

print(classification_report(y_true, y_pred))
print()
print("f1_score", f1_score(y_true, y_pred))


logits = np.load(os.path.join(PATH_EXPERIMENT_RESULTS, "dev_cnc.npy"))
y_prob = softmax(logits, axis=1)
y_pred = [1 if p[1] > 0.5 else 0 for p in y_prob]

print(classification_report(y_true, y_pred))
print()
print("f1_score", f1_score(y_true, y_pred))
