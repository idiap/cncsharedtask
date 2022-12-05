#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License


from __future__ import print_function

import datetime
import logging
import logging.config
import os
import random
import socket
import sys
import unicodedata
import zipfile
from pathlib import Path
from urllib import request

import numpy as np
import progressbar
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

logger = logging.getLogger(__name__)


def unicode_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize("NFD", text)


def print_eval_stats(predictions, labels):
    cm = confusion_matrix(labels, predictions, normalize="true")
    # precision = precision_score(labels, predictions)
    # recall = recall_score(labels, predictions)
    # logger.info(f"Precision: {precision}, Recall: {recall}")
    labels = ["GS", "GR", "G?"]
    preds = ["PS", "PR", "P?"]

    cm_string = f"    {'   '.join(preds)}\n" + "\n".join(
        [f"{l} {' '.join([f'{v:1.2f}' for v in row])}" for l, row in zip(labels, cm)]
    )
    logger.info(f"\nConfusion Matrix\n{cm_string}")


# Taken from https://stackoverflow.com/a/53643011
class DownloadProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def arg_topk(l, k):
    """
    Returns list of sorted indices  for top-k largest elements of the list l using numpy
    """
    ind = np.argpartition(l, -k)[-k:]
    return ind[np.argsort(l[ind])[::-1]]


def download_item(url, file_path):
    if not os.path.exists(file_path):
        logger.info(f"Downloading {file_path} from {url}")
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        try:
            request.urlretrieve(url, file_path, DownloadProgressBar())
        except ValueError:
            # Try prepending http, this is a common mistake
            request.urlretrieve("http://" + url, file_path, DownloadProgressBar())


def unzip(m):
    with zipfile.ZipFile(m, "r") as zf:
        target_path = os.path.dirname(m)
        for member in tqdm(zf.infolist(), desc=f"Unzipping {m} into {target_path}"):
            zf.extract(member, target_path)


def lazy_unzip(pathToZipFile: str):
    """
    Unzip pathToZipFile file, if pathToZipFile[:-len(".zip")] (unzipped) does not already exists.

    :param pathToZipFile: Path to zip file that should be unziped.
    :type pathToZipFile: str
    """

    if not os.path.isfile(pathToZipFile[:-4]):
        unzip(pathToZipFile)


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    if sll < 1:
        return results
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)


def get_device(t):
    return t.get_device() if t.get_device() > -1 else torch.device("cpu")


def touch(f):
    """
    Create empty file at given location f
    :param f: path to file
    """
    basedir = os.path.dirname(f)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    open(f, "a").close()


class LevelOnly(object):
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }

    def __init__(self, level):
        self.__level = self.levels[level]

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)


def count_lines(preprocessed_f):
    with open(preprocessed_f, encoding="utf-8") as f:
        return sum(1 for _ in f)


def get_model(m):
    if type(m) == DataParallel or type(m) == DistributedDataParallel:
        return m.module
    return m


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def deduplicate_list(l):
    new_l = []
    s = set()
    for x in l:
        if x in s:
            continue
        new_l.append(x)
        s.add(x)
    return new_l


def mkdir(s):
    if not os.path.exists(s):
        os.makedirs(s)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sum_parameters(model):
    return sum(p.view(-1).sum() for p in model.parameters() if p.requires_grad)


def report_parameters(model):
    num_pars = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    num_sizes = {
        name: p.shape for name, p in model.named_parameters() if p.requires_grad
    }
    return num_pars, num_sizes


def cat_lists(l):
    return [j for i in l for j in i]


def setup_logging(
    module, env_key="LOG_CFG", logpath=os.getcwd(), extra_name="", config_path=None
):
    """
    Setup logging configuration\n
    Logging configuration should be available in `YAML` file described by `env_key` environment variable

    :param module:     name of the module
    :param logpath:    path to logging folder [default: script's working directory]
    :param config_path: configuration file, has more priority than configuration file obtained via `env_key`
    :param env_key:    evironment variable containing path to configuration file
    :param default_level: default logging level, (in case of no local configuration is found)
    """

    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = timestamp + "_" + socket.gethostname() + "_" + extra_name

    path = config_path if config_path is not None else os.getenv(env_key, None)
    if path is not None and os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
            logfolder = os.path.join(module, stamp)
            for h in config["handlers"].values():
                if h["class"] == "logging.FileHandler":
                    h["filename"] = os.path.join(logpath, logfolder, h["filename"])
                    touch(h["filename"])
            for f in config["filters"].values():
                if "()" in f:
                    f["()"] = globals()[f["()"]]
            logging.config.dictConfig(config)
        return logfolder
    else:
        raise NotImplementedError("Missing logging configuration!")


def get_samples_cumsum(p, N=1):
    """
    :param p: probability distribution
    :param N: number of samples to draw
    """
    # renormalize, if needed
    if not np.allclose(p.sum(), 1.0):
        p = p / p.sum()

    # samples categorical with repetition
    cs = np.cumsum(p)
    try:
        return cs.searchsorted(np.random.uniform(0, cs[-1], size=N))
    except OverflowError as e:
        logger.error(f"p: {p}, N: {N}")
        raise e
