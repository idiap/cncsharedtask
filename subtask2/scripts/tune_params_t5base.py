#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License


import json
import logging
import math
import os
import pickle
import socket
import sys
import traceback

import numpy as np
import torch
import torch.multiprocessing as torch_mp
from hyperopt import STATUS_FAIL, fmin, hp, tpe
from hyperopt.exceptions import AllTrialsFailed
from hyperopt.mongoexp import MongoTrials

from subtask2.scripts.run_t5 import ddp_ce_extractor_fit
from subtask2.trainer.t5trainer import T5Trainer
from subtask2.utils.utility import setup_logging

logger = logging.getLogger(__name__)
SERVER = "YOUR-DB-SERVER-ADDRESS"
DB_ADDRESS = f"mongo://{SERVER}:1234/ce_t5_db/jobs"
DB_KEY = "ce_t5_base"


def obj(hpt_config):
    PATH = "YOUR-PATH-TO-REPOSITORY-ROOT"
    os.chdir(PATH)
    sys.path.append(PATH)

    log_folder = setup_logging(
        os.path.basename(sys.argv[0]).split(".")[0],
        logpath=".logs/",
        config_path="subtask2/configurations/logging.yml",
    )
    configuration = {
        "save_dir": ".saved",  # where the checkpoints will be saved
        "results": ".results",  # where validation results will be saved
        "data_cache_dir": "data/subtask2_preprocessed",  # where preprocessed data will be cached
        "transformers_cache": ".Transformers_cache",  # Where transformers download its models
        "output_predictions_wdir": ".predictions",
        "test_only": False,
        "max_prediction_iterations": 4,  # prevent model from predicting more than 4 annotations per sentence
        "validate_after_steps": 30,
        "train_data": "data/train_subtask2.csv",
        "val_data": "data/dev_subtask2.csv",
        "val_data_grouped": "data/dev_subtask2_grouped.csv",
        "tokenizer_type": "t5-base",
        "transformer_type": "t5-base",
        "save_threshold": 0.6,
        "avoid_f1_computation_when_loss_gt": 0.18,
        "max_grad_norm": 1.0,
        "weight_decay": "",
        "learning_rate": "",
        "adam_eps": 1e-08,
        "hidden_dropout": "",
        "attention_dropout": "",
        "optimizer": "adamw",
        "max_steps": 2_500,
        "scheduler": "",
        "scheduler_warmup_proportion": "",
        "patience": 100,
        "true_batch_size": "",
        "batch_size": 1,
        "fp16": False,
        "ddp": True,
        "ddp_backend": "nccl",
        "world_size": torch.cuda.device_count(),
        "max_num_threads": 36,
        "hpopt_tuning_mode": True,
    }
    configuration["batch_size"] = 1

    logger.info("Received config:")
    logger.info(json.dumps(hpt_config, indent=4, sort_keys=True))
    for key in hpt_config.keys():
        assert_msg = f"Missing key in hpt configuration: {key}"
        assert key in configuration, assert_msg
        assert configuration[key] == ""

    configuration.update(hpt_config)
    if configuration["scheduler"] is None:
        configuration["scheduler_warmup_proportion"] = None
    configuration["true_batch_size"] = int(configuration["true_batch_size"])
    for key in hpt_config.keys():
        assert configuration[key] != ""

    WORLD_SIZE = configuration["world_size"]

    torch_mp.spawn(
        ddp_ce_extractor_fit,
        args=(WORLD_SIZE, configuration, -1, log_folder, T5Trainer),
        nprocs=WORLD_SIZE,
        join=True,
    )
    try:
        fn = f"result_{socket.gethostname()}.pkl"
        with open(fn, "rb") as rf:
            result = pickle.load(rf)
        os.remove(fn)
    except BaseException as be:
        logger.error(be)
        logger.error(traceback.format_exc())
        result = {"loss": math.inf, "status": STATUS_FAIL}
    logger.info(f"Result: {result}")
    trials = MongoTrials(DB_ADDRESS, exp_key=DB_KEY)
    try:
        logger.info("Current best:")
        logger.info(trials.argmin)
    except AllTrialsFailed:
        logger.info("No successfull trials yet")
    return result


def run_hyperparam_opt():
    trials = MongoTrials(DB_ADDRESS, exp_key=DB_KEY)
    try:
        space = {
            "hidden_dropout": hp.uniform("hidden_dropout", low=0.0, high=0.5),
            "attention_dropout": hp.uniform("attention_dropout", low=0.0, high=0.5),
            "learning_rate": hp.loguniform(
                "learning_rate", low=np.log(1e-6), high=np.log(2e-4)
            ),
            "weight_decay": hp.uniform("weight_decay", low=0.0, high=3e-2),
            "true_batch_size": hp.quniform("true_batch_size", low=16, high=80, q=16),
            "scheduler_warmup_proportion": hp.uniform(
                "scheduler_warmup_proportion", low=0.0, high=0.2
            ),
            "scheduler": hp.choice("scheduler", ["linear", "constant", None]),
        }
        best = fmin(obj, space, trials=trials, algo=tpe.suggest, max_evals=1000)
        logger.info("#" * 20)
        logger.info(best)
    except KeyboardInterrupt as e:
        logger.info("INTERRUPTED")
        logger.info("#" * 20)
        logger.info(trials.argmin)


if __name__ == "__main__":
    log_folder = setup_logging(
        os.path.basename(sys.argv[0]).split(".")[0],
        logpath=".logs/",
        config_path="subtask2/configurations/logging.yml",
    )
    run_hyperparam_opt()
