#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def get_sentence(line):
    if line[1] is None or pd.isna(line[1]):
        return ""
    else:
        return line[1]


def load_datasets(data_dir, do_test=False, do_competition=False):
    dataset = {}
    splits = ["train", "dev"]
    if do_test:
        splits.append("test")
        if do_competition:
            splits.append("test_competition")

    for split in splits:
        filename = os.path.join(data_dir, f"{split}.csv")
        dataset[split] = pd.read_csv(filename, header=None).iloc[:, 0:2].values.tolist()

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Generate embeddings for test splits (test set is usually large, so we don't want to repeatedly generate embeddings for them)",
    )
    parser.add_argument(
        "--do_competition",
        action="store_true",
        help="Generate embeddings for official test set",
    )
    parser.add_argument(
        "--sbert_model",
        type=str,
        default="roberta-large",
        help="Sentence BERT model name",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/cnc", help="Path to few-shot data"
    )

    args = parser.parse_args()

    model = SentenceTransformer("{}-nli-stsb-mean-tokens".format(args.sbert_model))
    model = model.cuda()

    folder = args.data_dir
    dataset = load_datasets(
        folder, do_test=args.do_test, do_competition=args.do_competition
    )
    for split in dataset:
        print(split)
        lines = dataset[split]
        embeddings = []
        for line_id, line in tqdm(enumerate(lines)):
            sent = get_sentence(line)
            if line_id == 0:
                print("|", sent)
            emb = model.encode(sent)
            embeddings.append(emb)
        embeddings = np.stack(embeddings)
        np.save(
            os.path.join(folder, "{}_sbert-{}.npy".format(split, args.sbert_model)),
            embeddings,
        )


if __name__ == "__main__":
    main()
