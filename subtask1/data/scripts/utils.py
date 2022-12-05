#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

import os
from pathlib import Path


def make_dir(save_path=None, save_dir=None):
    if save_path is not None:
        path = Path(save_path)
        if not os.path.isdir(path.parent):
            os.makedirs(path.parent, exist_ok=True)
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
