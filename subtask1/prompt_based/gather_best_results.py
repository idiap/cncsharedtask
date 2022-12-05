#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

import os
import shutil
import sys

TOP_N_RESULTS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TARGET_METRIC = sys.argv[2] if len(sys.argv) > 2 else "f1_dev"
PATH_RESULTS = "result/"
PATH_BEST_RESULTS = "LOGS/BESTS/TOP-%d/" % TOP_N_RESULTS
COPY_FOLDERS = False

results = []
# filter_train_sizes = []
filter_train_sizes = [1000, 512]

for folder in os.listdir(PATH_RESULTS):
    if int(folder.split("-")[1]) in filter_train_sizes:
        continue

    try:
        with open(
            os.path.join(PATH_RESULTS, folder, "eval_results_cnc.txt"), "r"
        ) as reader:
            for line in reader.read().split("\n"):
                if "eval_f1" in line:
                    f1_eval = float(line.split(" = ")[1])

        with open(
            os.path.join(PATH_RESULTS, folder, "dev_results_cnc.txt"), "r"
        ) as reader:
            for line in reader.read().split("\n"):
                if "run.py" in line:
                    cmd = line
                elif "eval_f1" in line:
                    f1_dev = float(line.split(" = ")[1])

        results.append(
            {
                "path": os.path.join(PATH_RESULTS, folder),
                "cmd": cmd,
                "folder": folder,
                "f1_dev": f1_dev,
                "f1_eval": f1_eval,
                "f1_avg": (f1_dev + f1_eval) / 2,
                "f1_hm": 2 / (1 / f1_dev + 1 / f1_eval),
            }
        )
    except FileNotFoundError:
        pass

print("\n" + ("=" * 180))
print(TARGET_METRIC.upper())
for ix, result in enumerate(
    sorted(results, key=lambda t: -t[TARGET_METRIC])[:TOP_N_RESULTS]
):
    print(
        f'\n{ix + 1}) F1_dev: {result["f1_dev"]}; F1_eval: {result["f1_eval"]} ({result["folder"]})'
    )
    print("   COMMAND: " + result["cmd"])

    if COPY_FOLDERS:
        os.makedirs(PATH_BEST_RESULTS, exist_ok=True)
        shutil.copytree(
            result["path"],
            os.path.join(PATH_BEST_RESULTS, result["folder"]),
            dirs_exist_ok=True,
        )

# print("\nF1 Average")
# for result in sorted(results, key=lambda t: -t["f1_avg"])[:TOP_N_RESULTS]:
#     print(f'{result["folder"]}; F1_dev: {result["f1_dev"]}; F1_dev: {result["f1_eval"]};')
#     # print("\t" + result["cmd"])

# print("\nF1 Harmonic Mean")
# for result in sorted(results, key=lambda t: -t["f1_hm"])[:TOP_N_RESULTS]:
#     print(f'{result["folder"]}; F1_dev: {result["f1_dev"]}; F1_dev: {result["f1_eval"]};')
#     # print("\t" + result["cmd"])
