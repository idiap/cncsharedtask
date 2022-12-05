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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        type=str,
        help="A dictionary contains conditions that the experiment results need to fulfill (e.g., tag, task_name, few_shot_type)",
    )
    parser.add_argument("--template_dir", type=str, help="Template directory")
    parser.add_argument("--log", type=str, default="log", help="Log path.")

    args = parser.parse_args()

    condition = eval(args.condition)

    # default metric if not given is f1-score
    if "metric" not in condition or condition["metric"].startswith("f1"):
        key = "dev_eval_f1"
    elif condition["metric"].startswith("acc"):
        key = "dev_eval_acc"

    with open(args.log) as f:
        result_list = []
        for line in f:
            result_list.append(eval(line))

    seed_result = {}
    seed_result_template_id = {}  # avoid duplication

    for item in result_list:
        ok = True
        for cond in condition:
            if cond not in item or item[cond] != condition[cond]:
                ok = False
                break

        if ok:
            seed = item["seed"]
            if seed not in seed_result:
                seed_result[seed] = [item]
                seed_result_template_id[seed] = {item["template_id"]: 1}
            else:
                if item["template_id"] not in seed_result_template_id[seed]:
                    seed_result[seed].append(item)
                    seed_result_template_id[seed][item["template_id"]] = 1

    for seed in seed_result:
        print("Seed %d has %d results" % (seed, len(seed_result[seed])))

        # Load all templates
        with open(
            os.path.join(args.template_dir, "autotemplates-{}.txt".format(seed))
        ) as f:
            templates = []
            for line in f:
                templates.append(line.strip())

        # Write sorted templates
        fsort = open(
            os.path.join(args.template_dir, "autotemplates-{}.sort.txt".format(seed)),
            "w",
        )
        fscore = open(
            os.path.join(args.template_dir, "autotemplates-{}.score.txt".format(seed)),
            "w",
        )

        seed_result[seed].sort(key=lambda x: x[key], reverse=True)
        for item in seed_result[seed]:
            fsort.write(templates[item["template_id"]] + "\n")
            fscore.write("%.5f %s\n" % (item[key], templates[item["template_id"]]))


if __name__ == "__main__":
    main()
