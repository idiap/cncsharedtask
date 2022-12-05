#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

""" Script to get the results of the ensemble """

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
from sklearn import metrics
from torch.nn import functional as F


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of runs to perform greedy ensemble",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="deberta-v3-base",
        help="Name of the model in the seed folders. It could be '*' for several models",
    )

    parser.add_argument(
        "--ensemble-folder",
        type=str,
        default="./output/ensemble",
        help="Folder where the models are",
    )
    parser.add_argument(
        "--out-folder",
        type=str,
        default="./output/ensemble/results",
        help="Folder to store results",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="./output/labels/labels.txt",
        help="Labels for subtask 1",
    )

    return parser.parse_args()


def find_best_ensemble_greedy(
    base_folder="output",
    label_text="output/labels/labels.txt",
    models="deberta-v3-base",
):

    """function to find the best ensemble in a greedy way.
    models = you can pass '*' to use several models
    """

    # get the evaluations files per folder
    files = sorted(
        glob.glob(f"{base_folder}/*/{models}/evaluations/*.npy", recursive=True)
    )
    result_files = [f for f in files if "dev_" in f and f.endswith("npy")]

    # load labels and results per model,
    labels = np.loadtxt(label_text, dtype=int)
    result_matrices = {
        result_file: np.load(result_file) for result_file in result_files
    }

    # create the ensemble vars
    allm = set(result_matrices.keys())
    available_matrices = allm.copy()
    ensemble_matrices = set()

    seed = random.choice(list(available_matrices))

    # define a seed model and get F1
    best_F1 = find_F1(labels, [result_matrices[seed]])

    # while loop ntimes, n models
    while len(available_matrices) > 0:

        available_matrices = allm - ensemble_matrices
        found = False
        topickfrom = list(available_matrices)
        random.shuffle(topickfrom)

        # iterate over
        for m in topickfrom:
            candidate_matrices = ensemble_matrices.copy()
            candidate_matrices.add(m)
            F1 = find_F1(labels, [result_matrices[a] for a in candidate_matrices])

            # check if new F1 is better, otherwise continue
            if F1 > best_F1:
                best_F1 = F1
                ensemble_matrices.add(m)
                found = True
                break
        if not found:
            break

    return best_F1, ensemble_matrices


def find_F1(labels, results, use_softmax=True):
    """Function that computes the F1 score given the labels and outputs"""

    results = torch.Tensor(np.array(list(results))).cuda()

    # Models x batch x classes
    if use_softmax:
        results = F.softmax(results, -1)

    results = torch.mean(results, 0)
    labels = torch.Tensor(labels).cuda().long()
    maxpreds, argmaxpreds = torch.max(results, dim=1)

    total_preds = list(argmaxpreds.cpu().numpy())
    total_labels = list(labels.cpu().numpy())

    # get F1,
    F1 = metrics.f1_score(total_labels, total_preds, average="macro")

    return F1


def load_and_eval(
    found_best_ensemble=None,
    base_folder="output",
    models="deberta-v3-base",
    prefix="dev_",
    input_ids_file=None,
    lossfunction=None,
    weights=None,
    produce_result=False,
    calculate_f1=False,
    result_name="answer.json",
    strategy="average_logits",
):
    """Function to load an ensemble of models and evaluate in a given file
    prefix: either dev_ or test_ (inference files are encoded like that)
    """

    if found_best_ensemble == None:
        print("you need to pass a list with ensamble to load and eval")
        return

    # collect all the files and only select the top found by the greedy algorithm
    files = sorted(
        glob.glob(f"{base_folder}/*/{models}/evaluations/*.npy", recursive=True)
    )

    valid = [f for f in files if prefix in f and f.endswith("npy")]

    # extract only the path and change the prefix
    valid_ensemble_subset = [
        s.replace("dev_", prefix) for s in list(found_best_ensemble)
    ]
    result_files = [f for f in valid if prefix in f and f in valid_ensemble_subset]

    #  print(f"Ensemble of {len(result_files)} files")

    if produce_result and calculate_f1:
        labels = np.loadtxt(label_txt, dtype=int)
        labels = torch.Tensor(labels).cuda().long()

    # load the matrices to evaluate
    result_matrices = [np.load(result_file) for result_file in result_files]
    results = np.array(result_matrices)

    if strategy == "average_logits":
        feats = np.average(results, 0)
        results = torch.Tensor(feats).cuda()
    elif strategy == "sum_softmaxes":
        results = torch.Tensor(results).cuda()
        # Models x batch x classes
        results = F.softmax(results, -1)
        results = torch.sum(results, 0)
    elif strategy == "weighted_softmax_sum":
        results = torch.Tensor(results).cuda()
        # Models x batch x classes
        results = F.softmax(results, -1)
        for k in range(results.shape[0]):
            results[k] = results[k] * weights[k]
        results = torch.sum(results, 0)
    elif strategy == "avg_softmaxes":
        # Models x batch x classes
        results = torch.Tensor(results).cuda()
        results = F.softmax(results, -1)
        results = torch.mean(results, 0)

    else:
        return

    # get the final predictions
    softmaxed_results = (
        results if strategy == "avg_softmaxes" else F.softmax(results, -1)
    )
    maxpreds, argmaxpreds = torch.max(softmaxed_results, dim=1)
    total_preds = list(argmaxpreds.cpu().numpy())

    if not produce_result and calculate_f1:
        loss = lossfunction(results, labels)
        dev_loss = loss.item()
        total_labels = list(labels.cpu().numpy())

        correct_vec = argmaxpreds == labels
        total_correct = torch.sum(correct_vec).item()

        loss, acc = dev_loss, total_correct / results.shape[0]
        F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        F1_cls = metrics.f1_score(total_labels, total_preds, average=None)
        return loss, acc, F1, tuple(F1_cls)
    else:

        dirname = os.path.dirname(input_ids_file)
        # collect the ids to format the output as CNC challenge format
        ids_str = open(f"{dirname}/{prefix}ids.txt", "r").read().split()
        # collect the output in data
        data = []
        for ix, p in zip(ids_str, total_preds):
            data.append({"index": ix, "prediction": int(p)})

        with open(result_name, "w") as answer_file:
            answer_file.write("\n".join(json.dumps(i) for i in data))


def main():
    """reformat the input file into the CNC challenge format"""

    args = parse_args()
    output_folder = args.out_folder

    best_F1 = 0
    run = 0
    dictionary = {}
    # running the greddy ensemble for n num_runs
    for k in range(args.num_runs):
        # generate a random seed each iteration
        random.seed(k + random.randint(0, 1e10))

        # find the best ensemble
        F1, fs = find_best_ensemble_greedy(
            base_folder=args.ensemble_folder,
            label_text=args.labels_file,
            models=args.models,
        )
        # save logs in out folder
        if F1 > best_F1:
            best_F1 = F1

            # configure the output to dump in json
            str_id = f"run-{run}-f1-{F1:.3}-nmodels-{len(fs)}"
            dictionary[str_id] = {"f1-score": F1, "model": list(fs)}

            # log the f1 score and model list:
            with open(output_folder + "/logs", "a") as logs:
                json.dump(dictionary, logs, indent=4)

            print(f"run: {run}, F1-score: {F1}, number of models {len(fs)}")
        run += 1

    print(f"wrote output file with ensemble in: {output_folder}/logs")

    # After getting the ebst ensemble, get the JSON file in CNC format:
    # load and eval for dev set
    prefix = "dev_"
    print(f"generating outputs in CNC format in {output_folder}/{prefix}cnc.json")
    load_and_eval(
        found_best_ensemble=fs,
        base_folder=args.ensemble_folder,
        models=args.models,
        prefix=prefix,
        input_ids_file=f"{args.labels_file}",
        result_name=f"{output_folder}/{prefix}cnc.json",
    )

    # load and eval for test set
    prefix = "test_"
    print(f"generating outputs in CNC format in {output_folder}/{prefix}cnc.json")
    load_and_eval(
        found_best_ensemble=fs,
        base_folder=args.ensemble_folder,
        models=args.models,
        prefix=prefix,
        input_ids_file=f"{args.labels_file}",
        result_name=f"{output_folder}/{prefix}cnc.json",
    )


if __name__ == "__main__":
    main()
