#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License

# Trained using 24GB GPU

import logging
import os
import sys
import traceback
import uuid
from random import randint

import torch
import torch.multiprocessing as torch_mp

from subtask2.trainer.t5trainer import T5Trainer
from subtask2.utils.distributed_utils import cleanup_ddp, setup_ddp_envinit
from subtask2.utils.utility import mkdir, set_seed, setup_logging

config = {
    "save_dir": ".saved",  # where the checkpoints will be saved
    "results": ".results",  # where validation results will be saved
    "data_cache_dir": "data/subtask2_preprocessed",  # where preprocessed data will be cached
    "transformers_cache": ".Transformers_cache",  # Where transformers download its models
    "output_predictions_dir": ".predictions",
    "test_only": False,
    "max_prediction_iterations": 4,  # prevent model from predicting more than 4 annotations per sentence
    "validate_after_steps": 10,
    "train_data": "data/train_subtask2.csv",
    "val_data": "data/dev_subtask2.csv",
    "val_data_grouped": "data/dev_subtask2_grouped.csv",
    "tokenizer_type": "t5-large",
    "transformer_type": "t5-large",
    "save_threshold": 0.7,
    "avoid_f1_computation_when_loss_gt": 0.2,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "adam_eps": 1e-08,
    "optimizer": "adamw",
    "max_steps": 10000,
    "patience": 999,
    "batch_size": 1,
    "fp16": False,
    "ddp": True,
    "ddp_backend": "nccl",
    "world_size": torch.cuda.device_count(),
    "max_num_threads": 36,
}

best_config = {
    "learning_rate": 0.00016107176955317465,
    "hidden_dropout": 0.143607672048923,
    "attention_dropout": 0.4718606983478459,
    "weight_decay": 0.021490931909993172,
    "true_batch_size": 8,
    "scheduler_warmup_proportion": 0.15696656107839085,
    "scheduler": "constant",
}
config.update(best_config)


def ddp_ce_extractor_fit(rank, world_size, config, seed, log_folder, trainerclz):
    torch.set_num_threads(config["max_num_threads"])

    print(f"Running DDP process (PID {os.getpid()}) on rank {rank}.")
    if seed != -1:
        set_seed(seed)
    setup_logging(
        os.path.join(log_folder, "training", f"rank_{rank}"),
        logpath=".logs/",
        config_path="subtask2/configurations/logging.yml",
    )
    backend = config["ddp_backend"]

    original_bs = config["true_batch_size"]
    config["true_batch_size"] = int(config["true_batch_size"] / config["world_size"])
    logger.info(
        f"True batch size adjusted from {original_bs} to {config['true_batch_size']} due to"
        f" distributed world size {config['world_size']}"
    )

    setup_ddp_envinit(rank, world_size, backend, port=34526)
    try:
        if rank > -1:
            framework = trainerclz(config, local_rank=rank, global_rank=rank, seed=seed)
            framework.distributed = True
            framework.global_rank = rank
            framework.world_size = world_size
        else:
            framework = trainerclz(config, 0, 0, seed)
        validation_accuracy, model_path = framework.fit()
    except BaseException as be:
        logger.error(be)
        logger.error(traceback.format_exc())
        raise be
    finally:
        cleanup_ddp()


logger = logging.getLogger(__name__)


def run():
    # preprocess data
    setup_logging(
        os.path.basename(sys.argv[0]).split(".")[0],
        logpath=".logs/",
        config_path="subtask2/configurations/logging.yml",
    )

    framework = T5Trainer(config, 0, 0, 1234)
    framework.get_data(config)
    del framework

    mkdir(config["save_dir"])
    mkdir(config["results"])
    mkdir(config["data_cache_dir"])
    config["output_predictions_dir"] += f"_{uuid.uuid4().hex}"
    mkdir(config["output_predictions_dir"])

    RUNS = 10

    for i in range(RUNS):
        log_folder = setup_logging(
            os.path.basename(sys.argv[0]).split(".")[0],
            logpath=".logs/",
            config_path="subtask2/configurations/logging.yml",
        )
        seed = randint(0, 10_000)
        set_seed(seed)
        logger.info(f"Random seed: {seed}")
        WORLD_SIZE = config["world_size"]

        torch_mp.spawn(
            ddp_ce_extractor_fit,
            args=(WORLD_SIZE, config, seed, log_folder, T5Trainer),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == "__main__":
    run()
