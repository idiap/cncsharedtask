#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

"""Dataset utils for different data settings for GLUE."""

import logging
import os

import pandas as pd
from sklearn.metrics import f1_score
from transformers import DataProcessor, InputExample

logger = logging.getLogger(__name__)


class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets.
    """

    def __init__(self, data_dir):
        self.labels = None
        self.data_dir = data_dir

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self):
        """See base class."""
        examples = self._create_examples(
            pd.read_csv(
                os.path.join(self.data_dir, "train.csv"), header=None
            ).values.tolist(),
            "train",
        )
        return examples

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            pd.read_csv(
                os.path.join(self.data_dir, "dev.csv"), header=None
            ).values.tolist(),
            "dev",
        )

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            pd.read_csv(
                os.path.join(self.data_dir, "test.csv"), header=None
            ).values.tolist(),
            "test",
        )

    def get_test_competition_examples(
        self,
    ):
        """See base class."""
        return self._create_examples(
            pd.read_csv(
                os.path.join(self.data_dir, "test_competition.csv"), header=None
            ).values.tolist(),
            "test_competition",
        )

    def get_labels(self):
        """See base class."""
        if self.labels is None:
            self.labels = (
                pd.read_csv(os.path.join(self.data_dir, "train.csv"), header=None)[0]
                .unique()
                .tolist()
            )
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        examples_ids = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(
                    guid=line[2] if len(line) > 2 else guid,
                    text_a=line[1],
                    label=line[0],
                )
            )
            examples_ids.append(line[2])
        return examples


def text_classification_metrics(preds, labels):
    return {"acc": (preds == labels).mean()}


def acc_and_f1(preds, labels):
    acc = text_classification_metrics(preds, labels)["acc"]
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
