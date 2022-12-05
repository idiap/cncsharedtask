#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

""" Script to reformat input file to CNC challenge format """

import json
from sys import argv


def main():
    """reformat the input file into the CNC challenge format"""

    # this script always expect two files:
    file_path = argv[1]
    output_path = argv[2] if ".json" in argv[2] else argv[2] + ".json"

    # output path needs to be a JSON

    data = []
    with open(file_path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue  # column name
            line = line.rstrip().split("\t")
            data.append({"index": line[0], "prediction": int(line[1])})

    with open(output_path, "w") as fp:
        fp.write("\n".join(json.dumps(i) for i in data))


if __name__ == "__main__":
    main()
