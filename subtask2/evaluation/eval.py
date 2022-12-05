#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License

"""\
Official challenge evaluation script, with minor wrapper for ease of use
see original here https://github.com/tanfiona/CausalNewsCorpus/blob/master/evaluation/subtask2/evaluate.py
Edited by Martin Fajcik <mfajcik@idiap.ch>
Contains minor refactorizations compared to official version
"""

import itertools
import json
import os
import re
import socket
from ast import literal_eval

import pandas as pd
from datasets import load_metric


def get_combinations(list1, list2):
    return [
        list(zip(each_permutation, list2))
        for each_permutation in itertools.permutations(list1, len(list2))
    ]


def read_predictions(submission_file):
    predictions = []
    with open(submission_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                predictions.append(json.loads(line)["prediction"])
    return predictions


def clean_tok(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub("</*[A-Z]+\d*>", "", tok)


def get_BIO(text_w_pairs):
    tokens = []
    ce_tags = []
    next_tag = tag = "O"
    for tok in text_w_pairs.split(" "):

        # Replace if special
        if "<ARG0>" in tok:
            tok = re.sub("<ARG0>", "", tok)
            tag = "B-C"
            next_tag = "I-C"
        elif "</ARG0>" in tok:
            tok = re.sub("</ARG0>", "", tok)
            tag = "I-C"
            next_tag = "O"
        elif "<ARG1>" in tok:
            tok = re.sub("<ARG1>", "", tok)
            tag = "B-E"
            next_tag = "I-E"
        elif "</ARG1>" in tok:
            tok = re.sub("</ARG1>", "", tok)
            tag = "I-E"
            next_tag = "O"

        tokens.append(clean_tok(tok))
        ce_tags.append(tag)
        tag = next_tag

    return tokens, ce_tags


def get_BIO_sig(text_w_pairs):
    tokens = []
    s_tags = []
    next_tag = tag = "O"
    for tok in text_w_pairs.split(" "):
        # Replace if special
        if "<SIG" in tok:
            tok = re.sub("<SIG([A-Z]|\d)*>", "", tok)
            tag = "B-S"
            next_tag = "I-S"
            if "</SIG" in tok:  # one word only
                tok = re.sub("</SIG([A-Z]|\d)*>", "", tok)
                next_tag = "O"

        elif "</SIG" in tok:
            tok = re.sub("</SIG([A-Z]|\d)*>", "", tok)
            tag = "I-S"
            next_tag = "O"

        tokens.append(clean_tok(tok))
        s_tags.append(tag)
        tag = next_tag

    return tokens, s_tags


def get_BIO_all(text_w_pairs):
    tokens, ce_tags = get_BIO(text_w_pairs)
    tokens_s, s_tags = get_BIO_sig(text_w_pairs)
    failed_log = "Failed at: " + text_w_pairs
    assert tokens == tokens_s, failed_log
    assert len(ce_tags) == len(s_tags) == len(tokens)
    return tokens, ce_tags, s_tags


def format_results(ce_metric, sig_metric):
    final_results = {}
    metrics = ["precision", "recall", "f1", "accuracy"]
    final_results["Overall"] = {i: 0 for i in metrics}

    results = sig_metric.compute(zero_division=0)
    if "S" in results.keys():
        s_results = results["S"]
    else:
        s_results = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "number": 0}
    final_results["Signal"] = s_results
    final_results["Overall"]["accuracy"] += (
        results["overall_accuracy"] * s_results["number"]
    )
    accuracy_weight = s_results["number"]

    results = ce_metric.compute(zero_division=0)
    final_results["Cause"] = results["C"]
    final_results["Effect"] = results["E"]
    total_weight = 0
    for key in ["Cause", "Effect", "Signal"]:
        ddict = final_results[key]
        for i in metrics[:-1]:
            final_results["Overall"][i] += ddict[i] * ddict["number"]
        total_weight += ddict["number"]
    for k, v in final_results["Overall"].items():
        final_results["Overall"][k] = v / total_weight
    final_results["Overall"]["accuracy"] += (
        results["overall_accuracy"] * results["C"]["number"]
    )
    accuracy_weight += results["C"]["number"]
    final_results["Overall"]["accuracy"] /= accuracy_weight
    final_results["Overall"]["number"] = accuracy_weight

    return final_results


def keep_best_combinations_only(row, refs, preds):
    final_results = {}
    best_metric = -1

    for points in get_combinations(row.id, row.id):

        # initialise
        eid = socket.gethostname() + "_" + str(os.getpid())
        ce_metric = load_metric(
            "seqeval", cache_dir=".Transformers_cache", experiment_id=eid
        )
        sig_metric = load_metric(
            "seqeval", cache_dir=".Transformers_cache", experiment_id=eid
        )

        # add rounds
        for a, b in list(points):
            _, ce_ref, sig_ref = refs[a]
            _, ce_pred, sig_pred = preds[b]
            ce_metric.add(prediction=ce_pred, reference=ce_ref)
            sig_metric.add(prediction=sig_pred, reference=sig_ref)

        # compute
        results = format_results(ce_metric, sig_metric)
        key_metric = float(results["Overall"]["f1"])
        if key_metric > best_metric:
            # overwrite if best
            final_results = results
            best_metric = key_metric

    return final_results


def combine_dicts(d1, d2):
    the_keys = ["Cause", "Effect", "Signal"]
    metrics = ["precision", "recall", "f1"]
    d0 = {k: {i: 0 for i in metrics} for k in the_keys}

    for k in the_keys:
        total_weight = 0
        if k in d1.keys():
            for m in metrics:
                d0[k][m] += d1[k][m] * d1[k]["number"]
            total_weight += d1[k]["number"]
        if k in d2.keys():
            for m in metrics:
                d0[k][m] += d2[k][m] * d2[k]["number"]
            total_weight += d2[k]["number"]

        for m in metrics:
            d0[k][m] /= total_weight
            d0[k]["number"] = total_weight

    d0["Overall"] = {i: 0 for i in metrics}
    total_weight = 0
    for key in the_keys:
        ddict = d0[key]
        for i in metrics:
            d0["Overall"][i] += ddict[i] * ddict["number"]
        total_weight += ddict["number"]
    for k, v in d0["Overall"].items():
        d0["Overall"][k] = v / total_weight

    d0["Overall"]["number"] = d1["Overall"]["number"] + d2["Overall"]["number"]
    d0["Overall"]["accuracy"] = d1["Overall"]["accuracy"] * d1["Overall"]["number"]
    d0["Overall"]["accuracy"] += d2["Overall"]["accuracy"] * d2["Overall"]["number"]
    d0["Overall"]["accuracy"] /= d0["Overall"]["number"]

    return d0


def main(ref_df, pred_list, calculate_best_combi=True):
    # Convert
    ref_df, pred_list = keep_relevant_rows_and_unstack(ref_df, pred_list)
    assert len(pred_list) == len(ref_df)
    refs = [get_BIO_all(i) for i in ref_df["text_w_pairs"]]
    preds = [get_BIO_all(i) for i in pred_list]

    # Group
    if calculate_best_combi:
        grouped_df = ref_df.copy()
        grouped_df["id"] = [[i] for i in grouped_df.index]
        grouped_df = (
            grouped_df.groupby(["corpus", "doc_id", "sent_id"])[["eg_id", "id"]]
            .agg({"eg_id": "count", "id": "sum"})
            .reset_index()
        )
        grouped_df = grouped_df[grouped_df["eg_id"] > 1]
        req_combi_ids = [item for sublist in grouped_df["id"] for item in sublist]
    else:
        grouped_df = None
        req_combi_ids = []

    # For examples that DO NOT require combination search
    regular_ids = list(set(range(len(preds))) - set(req_combi_ids))

    eid = socket.gethostname() + "_" + str(os.getpid())
    ce_metric = load_metric(
        "seqeval", cache_dir=".Transformers_cache", experiment_id=eid
    )
    sig_metric = load_metric(
        "seqeval", cache_dir=".Transformers_cache", experiment_id=eid
    )

    for i in regular_ids:
        _, ce_ref, sig_ref = refs[i]
        _, ce_pred, sig_pred = preds[i]
        ce_metric.add(prediction=ce_pred, reference=ce_ref)
        sig_metric.add(prediction=sig_pred, reference=sig_ref)
    final_result = format_results(ce_metric, sig_metric)

    if grouped_df is not None:
        # For examples that require combination search
        for _, row in grouped_df.iterrows():
            best_results = keep_best_combinations_only(row, refs, preds)
            final_result = combine_dicts(final_result, best_results)

    return final_result


def keep_relevant_rows_and_unstack(ref_df, predictions):
    # Keep only causal examples
    predictions_w_true_labels = []
    eg_id_counter = []
    for i, row in ref_df.iterrows():
        if row.num_rs > 0:
            p = predictions[i]
            if len(p) > row.num_rs:
                # Note if you predict more than the number of relations we have, we only keep the first few.
                # We completely ignore the subsequent predictions.
                p = p[: row.num_rs]
            elif len(p) < row.num_rs:
                # Incorporate dummy predictions if there are insufficient predictions
                p.extend([row.text] * (row.num_rs - len(p)))
            predictions_w_true_labels.extend(p)
            eg_id_counter.extend(list(range(row.num_rs)))
    ref_df = ref_df[ref_df["num_rs"] > 0].reset_index(drop=True)

    # Expand into single rows
    ref_df["causal_text_w_pairs"] = ref_df["causal_text_w_pairs"].apply(
        lambda x: literal_eval(x)
    )
    ref_df = ref_df.explode("causal_text_w_pairs")
    ref_df = ref_df.rename(columns={"causal_text_w_pairs": "text_w_pairs"})
    ref_df["eg_id"] = eg_id_counter

    return ref_df.reset_index(drop=True), predictions_w_true_labels


def run_main_evalf1(input_dir, output_dir):
    submit_dir = os.path.join(input_dir, "res")
    truth_dir = os.path.join(input_dir, "ref")
    if not os.path.isdir(submit_dir):
        raise FileNotFoundError("{0} doesn't exist!".format(submit_dir))
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, "scores.txt")
        output_file = open(output_filename, "w")

        # read ground truth
        truth_file_name = os.listdir(truth_dir)[0]  # "truth.csv"
        truth_file = os.path.join(truth_dir, truth_file_name)
        truth = pd.read_csv(truth_file, sep=",", encoding="utf-8")

        # read predictions
        submit_file_name = os.listdir(submit_dir)[0]  # "submission.json"
        submission_answer_file = os.path.join(submit_dir, submit_file_name)
        predictions = read_predictions(submission_answer_file)

        if len(truth) != len(predictions):
            raise IndexError(
                "Number of entries in the submission.json do not match with the ground truth entry count!"
            )
        else:
            # evaluate
            final_results = main(truth, predictions, calculate_best_combi=True)
            # write to output file
            level_order = ["Overall", "Cause", "Effect", "Signal"]
            columns_order = ["Recall", "Precision", "F1", "Accuracy", "Number"]

            for level in level_order:
                ddict = final_results[level]
                for k in columns_order:
                    if k.lower() in ddict:
                        v = ddict[k.lower()]
                        if level == "Overall":
                            output_file.write("{0}:{1}\n".format(k.title(), v))
                        else:
                            output_file.write(
                                "{0}_{1}:{2}\n".format(level, k.title(), v)
                            )
                    else:
                        continue
            # output_file.write("{0}:{1}\n".format("MCC",0))

        output_file.close()
        return final_results


if __name__ == "__main__":
    # Set your prediction directory here
    DIR = ".predictions_final_0ff1245231354051a70ecda4946b7fe7"
    # Codalab Run
    INPUT_DIR = f"{DIR}/input_dir"
    OUTPUT_DIR = f"{DIR}/output_dir"
    r = run_main_evalf1(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    for k, v in r.items():
        for k2, v2 in v.items():
            r[k][k2] = float(r[k][k2])
    print(json.dumps(r, indent=4, sort_keys=True))
