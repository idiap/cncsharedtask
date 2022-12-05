#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

# Script to evaluate the Sequence Classification system

import argparse
import json
import os

import numpy as np
import torch
from datasets import Features, Value, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ft = {
    "index": Value("string"),
    "text": Value("string"),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--make-ensemble-outs",
        action="store_true",
        help="Whether to produce the outputs for ensemble models",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size you want to use for decoding",
    )

    parser.add_argument(
        "-m",
        "--input-model",
        required=True,
        help="Folder where the final model is stored",
    )
    parser.add_argument(
        "-i",
        "--input-files",
        required=True,
        help="String with paths to text or utt2spk_id files to be evaluated, it needs to match the 'test_names' variales",
    )
    parser.add_argument(
        "-n",
        "--test-names",
        required=True,
        help="Name of the test sets to be evaluated",
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        required=True,
        help="Folder where the final model is stored",
    )

    return parser.parse_args()


def batch(input_list, batch=1):
    """Iterator to go efficiently across the list of utterances to perform
    inference.
    input_list: input list with utterances
    batch: batch size
    """
    l = len(input_list)
    for ndx in range(0, l, batch):
        yield input_list[ndx : min(ndx + batch, l)]


def main():

    args = parse_args()
    # input model and some detailed outputs
    sec_classification_model = args.input_model
    path_to_files = args.input_files.rstrip().split(" ")
    test_set_names = args.test_names.rstrip().split(" ")
    output_folder = args.output_folder

    # evaluating that the number of test set names matches the amount of paths
    # passed:
    assert len(path_to_files) == len(
        test_set_names
    ), "you gave different number of paths and test sets"

    # create the output directory, in 'evaluations folder'
    os.makedirs(output_folder, exist_ok=True)

    print("\nLoading the sequence classification recognition model (speaker ID)\n")
    # Fetch the Model and tokenizer
    eval_model = AutoModelForSequenceClassification.from_pretrained(
        sec_classification_model
    )
    tokenizer = AutoTokenizer.from_pretrained(sec_classification_model)

    # get the labels of the model
    tag2id = eval_model.config.label2id
    id2tag = eval_model.config.id2label

    # Pipeline for sequence classification
    seq_cla = pipeline(
        "text-classification",
        model=eval_model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        top_k=1 if not args.make_ensemble_outs else 2,
        function_to_apply="softmax" if not args.make_ensemble_outs else None,
    )

    #  import ipdb; ipdb.set_trace()
    # main loop,
    for path_to_file, dataset_name in zip(path_to_files, test_set_names):

        print("         ******SEQUENCE CLASSIFICATION****** ")
        print(f"----    Evaluating dataset --> {dataset_name} -----")

        test_dataset = load_dataset(
            "csv", data_files=path_to_file, features=Features(ft), keep_in_memory=True
        )
        sequences = test_dataset["train"]["text"]
        ref_id = test_dataset["train"]["index"]

        print("Loaded dataset into memory")

        # run once the sequence classification system
        inference_out = []
        for x in batch(test_dataset["train"]["text"], args.batch_size):
            inference = seq_cla(x, batch_size=args.batch_size)
            inference_out += inference

        # either you make the submission for CNC or generate the logits for
        # ensemble experiments
        if args.make_ensemble_outs:
            # get the F1 score in the dev set:
            f1_score = json.load(open(f"{output_folder}/../all_results.json", "r"))[
                "eval_f1"
            ]
            path_to_output_file = (
                f"{output_folder}/{dataset_name}_f1_{f1_score:.4}_ensenmble"
            )

            data = []
            for reference, seqcla_output in zip(ref_id, inference_out):
                # Assemble the output one by one

                top1_score = seqcla_output[0]["score"]
                top2_score = seqcla_output[1]["score"]

                data.append(
                    [top1_score, top2_score]
                    if seqcla_output[0]["label"] == 0
                    else [top2_score, top1_score]
                )

            # conver to numpy array and save
            arr = np.asarray(data)
            np.save(path_to_output_file + ".npy", arr)

        else:
            # print the results in a txt file for submission
            path_to_output_file = f"{output_folder}/{dataset_name}_cnc_challenge.json"

            data = []
            for reference, seqcla_output in zip(ref_id, inference_out):
                # Assemble the output one by one
                # we just pick the first top value from the output
                label = f"{seqcla_output[0]['label']}"
                data.append({"index": reference, "prediction": int(label)})

            # print the inference file
            with open(path_to_output_file, "w") as fp:
                fp.write("\n".join(json.dumps(i) for i in data))

        print(
            f"done evaluating the model: {sec_classification_model} on {dataset_name}"
        )


if __name__ == "__main__":
    """Main code execution"""
    main()
