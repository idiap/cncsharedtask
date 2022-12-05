#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License


import csv
import json
import logging
import math
import os
import pickle
import shutil
import socket
import time
from collections import OrderedDict
from math import ceil

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from hyperopt import STATUS_OK
from jsonlines import jsonlines
from sklearn.metrics import f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from subtask2.dataset.get_best_f1 import get_bestF1_span
from subtask2.dataset.t5_ce_dataset import CSEDataset
from subtask2.evaluation.eval import run_main_evalf1
from subtask2.model.tokenizer_tools import init_tokenizer
from subtask2.utils.distributed_utils import share_list
from subtask2.utils.utility import (
    cat_lists,
    get_model,
    get_timestamp,
    mkdir,
    report_parameters,
    sum_parameters,
)

logger = logging.getLogger(__name__)


class T5Trainer:
    def __init__(self, config: dict, global_rank, local_rank, seed):
        """

        :param config: configuration of the training, as defined in the run script
        :param global_rank: global rank across ddp
        :param local_rank: rank on this machine
        :param seed: random seed to initialize with

        """
        # Note that multi-node multi-gpu training is unsupported so global_rank should be equal to local_rank here
        assert local_rank == global_rank, "Multi-node multi-gpu mode is unsupported"

        self.config = config
        self.global_rank = global_rank
        self.seed = seed
        self.DISTRIBUTED_RESULT_FN = f"t5trainer_result_S{self.seed}_r.pkl"
        self.best_score = config["save_threshold"]
        self.tokenizer = init_tokenizer(config)
        self.distributed = False
        self.last_ckpt_name = ""
        self.local_rank = local_rank
        self.torch_device = (
            local_rank if type(local_rank) != int else torch.device(local_rank)
        )
        self.no_improvement_rounds = 0
        self.world_size = "NODDP"

        # Token predicted, when no signal is present for this particular Cause and Effect for CES triplet
        self.EMPTY_TOKEN_ID = self.tokenizer.encode("▁empty", add_special_tokens=False)[
            0
        ]

    def init_model(self):
        trained_model = (
            torch.load(self.config["model_path"], map_location=torch.device("cpu"))
            if self.config["test_only"]
            else None
        )
        if trained_model is not None:
            assert self.config[
                "test_only"
            ], "Model initialization should be done only in test mode"
            config = trained_model.config.training_config
            logger.info(
                "Overwriting configuration. Initializing model with its own pre-training config"
            )
            logger.debug(
                f"Pretrained model's config:\n{json.dumps(config, indent=4, sort_keys=True)}"
            )
            config["test_only"] = True
        else:
            config = self.config
        cfg = T5Config.from_pretrained(
            config["transformer_type"], cache_dir=config["transformers_cache"]
        )
        cfg.gradient_checkpointing = config.get("gradient_checkpointing", False)
        cfg.attention_probs_dropout_prob = config["hidden_dropout"]
        cfg.hidden_dropout_prob = config["attention_dropout"]
        cfg.training_config = config

        if trained_model is not None:
            model = T5ForConditionalGeneration(config=cfg)
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                config["transformer_type"],
                config=cfg,
                cache_dir=config["transformers_cache"],
            )
        logger.info(f"Resizing token embeddings to length {len(self.tokenizer)}")
        model.resize_token_embeddings(len(self.tokenizer))
        model.training_steps = 0

        if trained_model is not None:
            # trained_model can be state_dict or pickled model
            if type(trained_model) == OrderedDict:
                model.load_state_dict(trained_model)
            else:
                model.load_state_dict(trained_model.state_dict())
                model.training_steps = trained_model.training_steps
                model.config.training_config = self.config
                if hasattr(trained_model, "optimizer_state_dict"):
                    model.optimizer_state_dict = trained_model.optimizer_state_dict
            del trained_model

        return self.make_parallel(model)

    def make_parallel(self, model):
        """
        Wrap model in DDP, if needed
        """

        model.to(self.local_rank)
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank])
            # Useful for debugging DDP
            # model = DDP(model, device_ids=[self.local_rank],find_unused_parameters=True)
            # torch.autograd.set_detect_anomaly(True)
        return model

    def get_data(self, config):
        if self.distributed:
            distributed_settings = {
                "rank": self.global_rank,
                "world_size": self.world_size,
            }
        else:
            distributed_settings = None
        train, val = None, None
        if not config["test_only"]:
            train = CSEDataset(
                config["train_data"],
                tokenizer=self.tokenizer,
                transformer=config["transformer_type"],
                cache_dir=self.config["data_cache_dir"],
                max_len=self.config.get("max_input_length", None),
                is_training=True,
                shuffle=True,
                reversed=self.config.get("ablation", "")
                == "reverse_cause_effect_genorder",
                skip_history=self.config.get("ablation", "")
                == "omit_history_from_inputs",
                distributed_settings=distributed_settings,
            )
        val = CSEDataset(
            config["val_data"],
            tokenizer=self.tokenizer,
            transformer=config["transformer_type"],
            cache_dir=self.config["data_cache_dir"],
            max_len=self.config.get("max_input_length", None),
            shuffle=self.config.get("shuffle_validation_set", False),
            is_training=False,
            reversed=self.config.get("ablation", "") == "reverse_cause_effect_genorder",
            skip_history=self.config.get("ablation", "") == "omit_history_from_inputs",
            distributed_settings=distributed_settings,
        )
        return train, val

    def fit(self):
        config = self.config

        logger.debug(json.dumps(config, indent=4, sort_keys=True))

        data = self.get_data(config)
        if data[0] is not None:
            train, val = data
        else:
            train, val = None, data[-1]

        if not config["test_only"]:
            logger.info(f"Training data examples:{len(train)}")

        logger.info(f"Validation data examples:{len(val)}")

        train_iter = None
        if train is not None:
            train_iter = DataLoader(
                train,
                batch_size=self.config["batch_size"],
                shuffle=False,
                pin_memory=True,
                prefetch_factor=2,
                collate_fn=CSEDataset.create_collate_fn(
                    pad_t=self.tokenizer.pad_token_id
                ),
            )
        val_iter_loss = DataLoader(
            val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=CSEDataset.create_collate_fn(
                pad_t=self.tokenizer.pad_token_id, validate_loss=True
            ),
        )
        val_iter_f1 = DataLoader(
            val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=CSEDataset.create_collate_fn(
                pad_t=self.tokenizer.pad_token_id, validate_F1=True
            ),
        )
        logger.info("Loading model...")
        model = self.init_model()

        logger.info(f"Trainable parameter checksum: {sum_parameters(model)}")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logger.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

        if not config["test_only"]:
            # Init optimizer
            # do not weight-decay biases and ln params
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": self.config["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if config["optimizer"] == "adamw":
                optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config["learning_rate"],
                    eps=self.config["adam_eps"],
                )
            elif config["optimizer"] == "adam":
                optimizer = Adam(
                    optimizer_grouped_parameters,
                    lr=self.config["learning_rate"],
                    eps=self.config["adam_eps"],
                )
            else:
                raise ValueError("Unsupported optimizer")

            if config.get("resume_checkpoint", False):
                optimizer.load_state_dict(get_model(model).optimizer_state_dict)

            # Init scheduler
            if "warmup_steps" in self.config:  # THIS WAS A TYPO IN MANY EXPERIMENTS
                self.config["scheduler_warmup_steps"] = self.config["warmup_steps"]
            if (
                "warmup_proportion" in self.config
            ):  # THIS WAS A TYPO IN MANY EXPERIMENTS
                self.config["scheduler_warmup_proportion"] = self.config[
                    "warmup_proportion"
                ]
            if self.config.get("scheduler", None) and (
                "scheduler_warmup_steps" in self.config
                or "scheduler_warmup_proportion" in self.config
            ):
                logger.info("Scheduler active!!!")
                t_total = self.config["max_steps"]
                warmup_steps = (
                    round(self.config["scheduler_warmup_proportion"] * t_total)
                    if "scheduler_warmup_proportion" in self.config
                    else self.config["scheduler_warmup_steps"]
                )
                scheduler = self.init_scheduler(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=t_total,
                    last_step=get_model(model).training_steps - 1,
                )
                logger.info(
                    f"Scheduler: warmup steps: {warmup_steps}, total_steps: {t_total}"
                )
            else:
                scheduler = None
        if not config["test_only"]:
            start_time = time.time()
            try:
                it = 0
                while get_model(model).training_steps < self.config[
                    "max_steps"
                ] and not self.no_improvement_rounds > self.config.get(
                    "patience", 9_999
                ):
                    if self.global_rank == 0:
                        logger.info(f"Epoch {it}")
                    train_loss = self.train_epoch(
                        model=model,
                        data_iter=train_iter,
                        val_loss_iter=val_iter_loss,
                        val_f1_iter=val_iter_f1,
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                    )
                    if math.isnan(train_loss):
                        logger.info("INF LOSS! Exiting...")
                        break
                    logger.info(
                        f"Global Rank {self.global_rank}: Training loss: {train_loss:.5f}"
                    )
                    it += 1
            except KeyboardInterrupt:
                logger.info("-" * 120)
                logger.info("Exit from training early.")
            finally:
                logger.info(
                    f"Finished after {(time.time() - start_time) / 60} minutes."
                )
                logger.info(f"Best performance: {self.best_score:.5f}")
                logger.info(f"Last checkpoint name: {self.last_ckpt_name}")
                if (
                    self.config.get("hpopt_tuning_mode", False)
                    and self.global_rank == 0
                ):
                    result = {"loss": -self.best_score, "status": STATUS_OK}
                    with open(f"result_{socket.gethostname()}.pkl", "wb") as wf:
                        pickle.dump(result, wf)
        else:
            loss, f1, _ = self.validate(
                model, val_loss_iter=val_iter_loss, val_f1_iter=val_iter_f1
            )
            logger.info(f"Evaluation Loss: {loss:.2f}, F1: {f1:.5f}")
        logger.info("#" * 50)
        return self.best_score, self.last_ckpt_name

    def init_scheduler(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_step: int = -1,
    ) -> LambdaLR:
        """
        Initialization of lr scheduler.

        :param last_step:
        :param num_training_steps:
        :param optimizer: The optimizer that is used for the training.
        :type optimizer: Optimizer
        :return: Created scheduler.
        :rtype: LambdaLR
        """
        if last_step > 0:
            logger.info(f"Setting scheduler step to {last_step}")
            # We need initial_lr, because scheduler demands it.
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

        if self.config["scheduler"] == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=last_step,
            )
        elif self.config["scheduler"] == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=0.5,
                last_epoch=last_step,
            )
        elif self.config["scheduler"] == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                last_epoch=last_step,
            )
        else:
            scheduler = None

        return scheduler

    def forward_pass(self, *args, **kwargs):
        """
        FP16 wrapper
        """
        if self.config["fp16"]:
            with torch.cuda.amp.autocast():
                return self._forward_pass(*args, **kwargs)
        else:
            return self._forward_pass(*args, **kwargs)

    def _forward_pass(self, src, src_mask, tgt, tgt_mask, metadata=None, model=None):
        """

        :param src:
        :param src_mask:
        :param tgt:
        :param tgt_mask:
        :param metadata:
        :param model:
        """
        outputs = model(
            input_ids=src,
            attention_mask=src_mask,
            decoder_input_ids=tgt[:, :-1].contiguous(),
            decoder_attention_mask=tgt_mask[:, :-1].contiguous(),
            use_cache=False,
        )
        labels = tgt[:, 1:].reshape(-1)
        losses = F.cross_entropy(
            outputs.logits.view(-1, get_model(model).config.vocab_size),
            labels,
            reduction="none",
        ).view(tgt[:, 1:].shape)
        lm_loss = losses.mean(-1).mean(-1)
        validation_outputs = {
            "metadata": metadata,
        }
        if self.config.get("eval_signal_accuracies", False):
            # 0 is '_signal', 1 is ':', 2 is _empty or smt else
            empty_signal_probs = F.softmax(outputs.logits[2::3][:, 2], -1)[
                :, self.EMPTY_TOKEN_ID
            ]
            validation_outputs["empty_preds"] = (
                (empty_signal_probs > 0.5).cpu().tolist()
            )
        return lm_loss, validation_outputs

    def train_epoch(
        self,
        model,
        data_iter,
        val_loss_iter,
        val_f1_iter,
        optimizer: Optimizer,
        lr_scheduler: LambdaLR,
    ):
        """

        :param model:
        :param data_iter:
        :param val_iter:
        :param optimizer:
        :param lr_scheduler:
        :return:
        """
        #  Training flags
        model.train()
        # Make sure parameters are zero
        optimizer.zero_grad()

        # Determine update ratio, e.g. if true_batch_size = 32 and batch_size=8, then
        # gradients should be updated  every 4th iteration (except for last update!)
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        err = f"True batch size {self.config['true_batch_size']} is not divisible batch size {self.config['batch_size']}"
        assert self.config["true_batch_size"] % self.config["batch_size"] == 0, err
        updated = False
        adjusted_for_last_update = (
            False  # In last update, the ba tch size is adjusted to whats left
        )

        # Calculate total number of updates per epoch
        total = ceil(len(data_iter.dataset) / data_iter.batch_size)

        # For progressive  training loss  reporting
        total_losses = []
        losses_per_update = []
        total_updates = 0

        # If we use fp16, gradients must be scaled
        grad_scaler = None
        if self.config["fp16"]:
            grad_scaler = torch.cuda.amp.GradScaler()

        # it = tqdm(data_iter, total=total)
        it = data_iter
        iteration = 0
        for src, src_mask, target, target_mask, metadata in it:

            iteration += 1
            updated = False

            src_shapes = src.shape
            src_mask_shapes = src_mask.shape
            tgt_shapes = target.shape
            tgt_mask_shapes = target_mask.shape
            try:
                # Useful for debugging
                # inps = [" ".join(self.tokenizer.convert_ids_to_tokens(inp)) for inp in src]
                # tars = [" ".join(self.tokenizer.convert_ids_to_tokens(inp)) for inp in target]
                # for s, t in zip(inps, tars):
                #     print(s + "\n" + t + "\n")
                # print("*" * 10)
                # Adjust update ratio for last update if needed
                if (
                    (total - iteration) < update_ratio
                    and len(losses_per_update) == 0
                    and not adjusted_for_last_update
                ):
                    update_ratio = total - iteration
                    adjusted_for_last_update = True
                if self.config.get("1sample_perbatch", False):
                    # Do forward-backward computation for 1 sample only, for e
                    losses = [
                        self.get_loss(
                            adjusted_for_last_update,
                            grad_scaler,
                            losses_per_update,
                            metadata,
                            model,
                            src_perex.unsqueeze(0),
                            src_mask_perex.unsqueeze(0),
                            target_perex.unsqueeze(0),
                            target_mask_perex.unsqueeze(0),
                            update_ratio * len(src),
                        ).item()
                        for src_perex, src_mask_perex, target_perex, target_mask_perex in zip(
                            src, src_mask, target, target_mask
                        )
                    ]
                    loss = sum(losses)
                else:
                    loss = self.get_loss(
                        adjusted_for_last_update,
                        grad_scaler,
                        losses_per_update,
                        metadata,
                        model,
                        src,
                        src_mask,
                        target,
                        target_mask,
                        update_ratio,
                    ).item()
                # record losses to list
                losses_per_update.append(loss)

                if (
                    len(losses_per_update) == update_ratio
                    and not adjusted_for_last_update
                ):
                    # check that the model is in training mode
                    assert model.training

                    # grad clipping should be applied to unscaled gradients
                    if self.config["fp16"]:
                        # Unscales the gradients of optimizer's assigned params in-place
                        grad_scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        self.config["max_grad_norm"],
                    )
                    # compute training loss
                    loss_per_update = sum(losses_per_update)
                    total_updates += 1

                    # if we train masker, get avg amount of masked tokens
                    logger.debug(
                        f"S: {get_model(model).training_steps + 1}, "
                        f"Loss: {loss_per_update}, "
                        f"EID: {[e['doc_id'] for e in metadata]}, "
                        f"training flag {model.training}"
                    )
                    total_losses += losses_per_update
                    losses_per_update = []

                    if self.config["fp16"]:
                        # Unscales gradients and calls
                        # or skips optimizer.step()
                        grad_scaler.step(optimizer)
                        # Updates the scale for next iteration
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    get_model(model).training_steps += 1
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    updated = True

                    # it.set_description(f"Steps: {get_model(model).training_steps}, "
                    #                    f"Training loss: {loss_per_update:.5f}")

                    # Validate after every validate_after_steps steps
                    if (
                        get_model(model).training_steps > 1
                        and get_model(model).training_steps
                        % self.config["validate_after_steps"]
                        == 0
                    ):
                        val_loss, val_f1, improvement = self.validate(
                            model,
                            val_loss_iter=val_loss_iter,
                            val_f1_iter=val_f1_iter,
                            optimizer_dict=optimizer.state_dict(),
                        )
                        model = model.train()

                        self.no_improvement_rounds = (
                            0 if improvement else self.no_improvement_rounds + 1
                        )

                    # Exit if maximal number of steps is reached, or patience is surpassed
                    if get_model(model).training_steps == self.config[
                        "max_steps"
                    ] or self.no_improvement_rounds > self.config.get(
                        "patience", 9_999
                    ):
                        break
            # Catch out-of-memory errors
            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    torch.cuda.empty_cache()
                    logger.error(str(e))
                    logger.error("OOM detected, emptying cache...")
                    logger.error(
                        f"src_shape {src_shapes}\n"
                        f"src_mask_shape{src_mask_shapes}\n"
                        f"tgt_shape {tgt_shapes}\n"
                        f"tgt_mask_shape{tgt_mask_shapes}\n"
                    )
                    time.sleep(3)
                else:
                    raise e
        if not updated:
            # logger.debug(f"I{iteration}_S{get_model(model).training_steps}, Doing last step update R{self.rank}"
            #               f"from {len(losses_per_update)} losses")
            # check that the model is in training mode
            assert model.training
            # Do the last step if needed
            if self.config["fp16"]:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                self.config["max_grad_norm"],
            )
            if self.config["fp16"]:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            get_model(model).training_steps += 1
            if lr_scheduler is not None:
                lr_scheduler.step()
            losses_per_update = []

            if (
                get_model(model).training_steps > 1
                and get_model(model).training_steps
                % self.config["validate_after_steps"]
                == 0
            ):
                val_loss, val_f1, improvement = self.validate(
                    model,
                    val_loss_iter=val_loss_iter,
                    val_f1_iter=val_f1_iter,
                    optimizer_dict=optimizer.state_dict(),
                )
                model = model.train()
                self.no_improvement_rounds = (
                    0 if improvement else self.no_improvement_rounds + 1
                )

        # Validate after epoch
        # self.validate(model, val_iter,
        #               optimizer_dict=optimizer.state_dict())
        return sum(total_losses) / total_updates

    def get_loss(
        self,
        adjusted_for_last_update,
        grad_scaler,
        losses_per_update,
        metadata,
        model,
        src,
        src_mask,
        target,
        target_mask,
        update_ratio,
    ):
        # Move to gpu
        src, src_mask = src.to(self.torch_device), src_mask.to(self.torch_device)
        target, target_mask = target.to(self.torch_device), target_mask.to(
            self.torch_device
        )

        # If DDP is active, synchronize gradients only in update step!
        # Do not do this in the last minibatch, if dataset is not divisible by minibatch (and thus there was no adjustment)
        if (
            self.distributed
            and self.world_size > 1
            and not (len(losses_per_update) + 1 == update_ratio)
            and not adjusted_for_last_update
        ):
            with model.no_sync():
                loss, validation_outputs = self.forward_pass(
                    src, src_mask, target, target_mask, metadata, model
                )
                loss /= update_ratio
                grad_scaler.scale(loss).backward() if self.config[
                    "fp16"
                ] else loss.backward()
        else:
            loss, validation_outputs = self.forward_pass(
                src, src_mask, target, target_mask, metadata, model
            )
            loss /= update_ratio
            grad_scaler.scale(loss).backward() if self.config[
                "fp16"
            ] else loss.backward()
        return loss

    @torch.no_grad()
    def validate(
        self, model, *args, val_loss_iter=None, val_f1_iter=None, optimizer_dict=None
    ):
        improvement = False
        validation_loss = self.validate_loss(model, *args, data_iter=val_loss_iter)
        if validation_loss <= self.config.get(
            "avoid_f1_computation_when_loss_gt", 999.0
        ) or (
            self.config["test_only"]
            and not self.config.get("eval_signal_accuracies", False)
        ):
            validation_f1 = self.validate_f1(model, *args, data_iter=val_f1_iter)
        else:
            validation_f1 = 0.0
            improvement = True

        if validation_f1 > self.best_score:
            self.best_score = validation_f1
            improvement = True
            if self.global_rank == 0 and not self.config["test_only"]:
                logger.info(f"NEW BEST ->>> Validation F1: {validation_f1:5f}")
                self.save_model(self.best_score, model, optimizer_dict)

        return validation_loss, validation_f1, improvement

    def save_model(self, score, model, optimizer_dict):
        serializable_model_name = self.config["transformer_type"].replace("/", "_")
        saveable_model = get_model(model)
        saveable_model.optimizer_state_dict = optimizer_dict

        # Note that model training is fully resumable
        # it contains .optimizer_state_dict and .training_steps (=number of updates)
        saved_name = os.path.join(
            self.config["save_dir"],
            f"CausalExtractor_"
            f"BS{score:.4f}_"
            f"S{get_model(model).training_steps}_"
            f"M{serializable_model_name}_"
            f"{get_timestamp()}_{socket.gethostname()}",
        )
        self.last_ckpt_name = saved_name
        # Deal with model-parallelism
        device_map = None
        if hasattr(saveable_model, "model_parallel") and saveable_model.model_parallel:
            device_map = saveable_model.device_map
            model.deparallelize()
        torch.save(saveable_model, saved_name)
        if device_map is not None:
            saveable_model.parallelize(device_map)

    def get_pred_for_prefix(
        self, prefix, model, src, src_mask, return_emptysignal_logitscore=False
    ):
        if len(src.shape) < 2:
            src.unsqueeze_(0)
        if len(src_mask.shape) < 2:
            src_mask.unsqueeze_(0)

        def eosindex(l):
            try:
                # find last eos
                return len(l) - 1 - l[::-1].index(self.tokenizer.eos_token_id)
            except:
                # find first non-starting pad
                try:
                    return l[1:].index(self.tokenizer.pad_token_id) + 1
                except:
                    # just return whole sequence
                    return len(l)

        # useful for understanding
        # inps = [" ".join(self.tokenizer.convert_ids_to_tokens(inp)) for inp in src]
        # decoder_inps = " ".join(self.tokenizer.convert_ids_to_tokens(prefix))
        output = get_model(model).generate(
            input_ids=src,
            attention_mask=src_mask,
            min_length=2 + len(prefix) - 1,  # answer should never be empty
            output_scores=True,
            return_dict_in_generate=True,
            decoder_start_token_id=prefix,
        )
        # useful for understanding
        # generated_outps = [" ".join(self.tokenizer.convert_ids_to_tokens(out)) for out in output.sequences]
        #
        # logger.debug(f"ENCINP: {inps}")
        # logger.debug(f"DECINP: {decoder_inps}")
        # logger.debug(f"GENOUT: {generated_outps}")
        #
        # logger.debug("*****" * 5)

        sequences = [
            s[len(prefix) : eosindex(s)] for s in output.sequences.cpu().tolist()
        ]
        if return_emptysignal_logitscore:
            logit_score = output.scores[0][:, self.EMPTY_TOKEN_ID].item()
            return (
                sequences,
                [
                    self.tokenizer.decode(sequence, skip_special_tokens=False)
                    for sequence in sequences
                ],
                logit_score,
            )
        # print(f"DEBUG: {self.tokenizer.decode(output.sequences[0], skip_special_tokens=False)}")
        return sequences, [
            self.tokenizer.decode(sequence, skip_special_tokens=False)
            for sequence in sequences
        ]

    @torch.no_grad()
    def validate_loss(self, model, data_iter):
        """
        :param model:
        :param data_iter:
        :return:
        """
        flag = model.training
        model = model.eval()

        # Remove gradients, to save memory
        for param in model.parameters():
            param.grad = None

        # Calculate total number of updates per epoch
        total = ceil(len(data_iter.dataset) / data_iter.batch_size)

        # For progressive  training loss  reporting
        total_losses = []

        # it = tqdm(data_iter, total=total)
        it = data_iter
        iteration = 0

        empty_preds = []
        empty_gts = []
        for src, src_mask, target, target_mask, metadata in it:
            iteration += 1
            src, src_mask = src.to(self.torch_device), src_mask.to(self.torch_device)
            target, target_mask = target.to(self.torch_device), target_mask.to(
                self.torch_device
            )
            loss, validation_outputs = self.forward_pass(
                src, src_mask, target, target_mask, metadata, model
            )
            total_losses.append(loss.item())
            if self.config.get("eval_signal_accuracies", False):
                empty_preds += validation_outputs["empty_preds"]
                # 0 is start-pad token, 1 is '_signal', 2 is ':', 3 is _empty or smt else
                empty_gts += (target[2::3][:, 3] == self.EMPTY_TOKEN_ID).cpu().tolist()

        if self.config.get("eval_signal_accuracies", False):
            hits = [p == g for p, g in zip(empty_preds, empty_gts)]
            logger.info(
                f"Signal accuracy: {sum(hits) / len(empty_gts) * 100:.2f} ({sum(hits)}/{len(empty_gts)})"
            )
            logger.info(
                f"Empty signal F1: {f1_score(y_true=empty_gts, y_pred=empty_preds):.2f}"
            )
            logger.info(
                f"Total relations with signal {sum([1 - e for e in empty_gts])}"
            )
        if flag:
            model.train()
        if self.world_size == "NODDP":
            dist_losslist = total_losses
        else:
            dist_losslist = cat_lists(
                share_list(
                    total_losses, rank=self.global_rank, world_size=self.world_size
                )
            )

        val_loss = sum(dist_losslist) / len(dist_losslist)
        if self.global_rank == 0:
            logger.info(f"Validation loss: {val_loss:5f}")

        if self.world_size != "NODDP":
            dist.barrier()

        return val_loss

    @torch.no_grad()
    def validate_f1(self, model, data_iter):

        tokenizer_kwargs = {"add_special_tokens": False, "truncation": False}
        pp = lambda x: torch.LongTensor(x).to(self.torch_device)
        cause_prefix = pp(
            [self.tokenizer.pad_token_id]
            + self.tokenizer.encode("cause: ", **tokenizer_kwargs)
        )
        effect_prefix = pp(
            [self.tokenizer.pad_token_id]
            + self.tokenizer.encode("effect: ", **tokenizer_kwargs)
        )
        signal_prefix = pp(
            [self.tokenizer.pad_token_id]
            + self.tokenizer.encode("signal: ", **tokenizer_kwargs)
        )

        flag = model.training
        model = model.eval()

        # Remove gradients, to save memory
        for param in model.parameters():
            param.grad = None

        # Calculate total number of updates per epoch
        total = ceil(len(data_iter.dataset) / data_iter.batch_size)

        # For progressive  training loss  reporting
        predictions = []

        it = data_iter
        iteration = 0

        def pad_sequences(sequences):
            sequence_lengths = [len(seq) for seq in sequences]
            max_length = max(sequence_lengths)
            padded_sequences = [
                seq + [self.tokenizer.pad_token_id] * (max_length - len(seq))
                for seq in sequences
            ]
            return padded_sequences

        pp = lambda x: torch.LongTensor(pad_sequences(x)).to(self.torch_device)
        max_length = data_iter.dataset.max_len

        for src, src_mask, target, target_mask, metadata in it:
            iteration += 1
            src, src_mask = src.to(self.torch_device), src_mask.to(self.torch_device)
            history = [[] for _ in src]
            for k in range(self.config["max_prediction_iterations"]):
                if (
                    k > 0
                    and self.config.get("ablation", "") == "omit_history_from_inputs"
                ):
                    # in case there is no history ablation, just predict N times the same thing
                    for i in range(len(history)):
                        assert len(history[i]) == 1
                        history[i] += history[i] * (
                            self.config["max_prediction_iterations"] - 1
                        )
                    break

                # prepare history for k-th round
                history_inputs = [
                    CSEDataset.assemble_history_inputs(h, self.tokenizer)
                    for h in history
                ]

                if self.config.get("ablation", "") == "reverse_cause_effect_genorder":
                    (
                        predicted_causes,
                        predicted_causes_tokens,
                        predicted_effects,
                        predicted_effects_tokens,
                    ) = self.generate_effect_then_cause(
                        cause_prefix,
                        effect_prefix,
                        history_inputs,
                        k,
                        max_length,
                        metadata,
                        model,
                        pp,
                        src,
                        src_mask,
                    )
                else:
                    (
                        predicted_causes,
                        predicted_causes_tokens,
                        predicted_effects,
                        predicted_effects_tokens,
                    ) = self.generate_cause_then_effect(
                        cause_prefix,
                        effect_prefix,
                        history_inputs,
                        k,
                        max_length,
                        metadata,
                        model,
                        pp,
                        src,
                        src_mask,
                    )
                # prepare signal input
                inp_signal = []
                for m, predicted_cause, predicted_effect, history_inputs_perex in zip(
                    metadata, predicted_causes, predicted_effects, history_inputs
                ):
                    sentence_seq = CSEDataset.assemble_input_for_signal(
                        predicted_causes, predicted_effect, m["text"], self.tokenizer
                    )
                    inp_signal_per_ex = sentence_seq + history_inputs_perex
                    if len(inp_signal_per_ex) > max_length:
                        inp_signal_per_ex = inp_signal_per_ex[:max_length]
                    inp_signal.append(inp_signal_per_ex)

                (
                    predicted_signals_tokens,
                    predicted_signals,
                    emptysignal_logit_score,
                ) = self.get_pred_for_prefix(
                    signal_prefix,
                    model,
                    pp(inp_signal),
                    pp([[1] * len(ie) for ie in inp_signal]),
                    return_emptysignal_logitscore=True,
                )
                for i in range(len(history)):
                    history[i].append(
                        {
                            "ces_triplet": {
                                "cause": predicted_causes[i],
                                "cause_ids": predicted_causes_tokens[i],
                                "effect": predicted_effects[i],
                                "effect_ids": predicted_effects_tokens[i],
                                "signal": predicted_signals[i],
                                "signal_ids": predicted_signals_tokens[i],
                            },
                            "emptysignal_logit_score": emptysignal_logit_score,
                        }
                    )
            # useful for understanding
            # logger.info(json.dumps(history, indent=4, sort_keys=True))
            # logger.info("**" * 20)
            for m, h in zip(metadata, history):
                pred = {"sentence": m["text"], "id": m["index"], "predictions": h}
                if self.config.get("dump_validation_metadata", False):
                    pred[
                        "metadata"
                    ] = m  ##!!!! THESE ARE ONLY METADATA OF THE RANDOM CES TRIPLET FROM THE SENTENCE
                    pred["sentence_ids"] = self.tokenizer.encode_plus(
                        m["text"], return_offsets_mapping=True, add_special_tokens=False
                    )
                predictions.append(pred)
        if self.world_size == "NODDP":
            dist_predictions = predictions
        else:
            dist_predictions = cat_lists(
                share_list(
                    predictions, rank=self.global_rank, world_size=self.world_size
                )
            )

        def postprocess_predictions(dist_predictions, val_data_grouped):
            d = dict()
            for p in dist_predictions:
                sentence = p["sentence"]
                tokenized = self.tokenizer.encode_plus(
                    sentence, return_offsets_mapping=True, add_special_tokens=False
                )
                modified_preds = []
                for pred in p["predictions"]:
                    ces = pred["ces_triplet"]

                    cause_span, cause_score = get_bestF1_span(
                        tokenized.input_ids, ces["cause_ids"]
                    )
                    effect_span, effect_score = get_bestF1_span(
                        tokenized.input_ids, ces["effect_ids"]
                    )

                    if ces["signal"] == "empty":
                        signal_span, signal_score = "", 0.0
                    else:
                        signal_span, signal_score = get_bestF1_span(
                            tokenized.input_ids, ces["signal_ids"]
                        )

                    offsets = np.array([[s, e] for s, e in tokenized.offset_mapping])
                    original_words = np.array([sentence[s:e] for s, e in offsets])

                    start_character = end_character = 99999
                    STARTTAG = "<ARG0>"
                    ENDTAG = "</ARG0>"
                    if not cause_score < 0.1:
                        start_character = tokenized.offset_mapping[cause_span[0]][0]
                        end_character = tokenized.offset_mapping[cause_span[1]][1]

                        # modify sentence
                        modified_sentence = (
                            sentence[:end_character] + ENDTAG + sentence[end_character:]
                        )
                        modified_sentence = (
                            modified_sentence[:start_character]
                            + STARTTAG
                            + modified_sentence[start_character:]
                        )
                        # modify offsets
                        offsets[:, 0][offsets[:, 0] > end_character] += len(ENDTAG)
                        offsets[:, 1][offsets[:, 1] > end_character] += len(ENDTAG)
                        offsets[:, 0][offsets[:, 0] >= start_character] += len(STARTTAG)
                        offsets[:, 1][offsets[:, 1] > start_character] += len(STARTTAG)
                        modified_offset_words = np.array(
                            [
                                modified_sentence[s:e]
                                .replace("</ARG0>", "")
                                .replace("<ARG0>", "")
                                for s, e in offsets
                            ]
                        )
                        assert (original_words == modified_offset_words).all()
                    else:
                        modified_sentence = sentence

                    if not effect_score < 0.1:
                        STARTTAG = "<ARG1>"
                        ENDTAG = "</ARG1>"

                        start_character = offsets[effect_span[0]][0]
                        end_character = offsets[effect_span[1]][1]

                        # modify sentence
                        modified_sentence = (
                            modified_sentence[:end_character]
                            + ENDTAG
                            + modified_sentence[end_character:]
                        )
                        modified_sentence = (
                            modified_sentence[:start_character]
                            + STARTTAG
                            + modified_sentence[start_character:]
                        )
                        # modify offsets
                        offsets[:, 0][offsets[:, 0] > end_character] += len(ENDTAG)
                        offsets[:, 1][offsets[:, 1] > end_character] += len(ENDTAG)
                        offsets[:, 0][offsets[:, 0] >= start_character] += len(STARTTAG)
                        offsets[:, 1][offsets[:, 1] > start_character] += len(STARTTAG)
                        modified_offset_words = np.array(
                            [
                                modified_sentence[s:e]
                                .replace("</ARG0>", "")
                                .replace("<ARG0>", "")
                                .replace("</ARG1>", "")
                                .replace("<ARG1>", "")
                                for s, e in offsets
                            ]
                        )
                        assert (original_words == modified_offset_words).all()
                    else:
                        start_character = end_character = 99999

                    if not signal_score < 0.1:
                        STARTTAG = "<SIG0>"
                        ENDTAG = "</SIG0>"
                        start_character = offsets[signal_span[0]][0]
                        end_character = offsets[signal_span[1]][1]

                        # modify sentence
                        modified_sentence = (
                            modified_sentence[:end_character]
                            + ENDTAG
                            + modified_sentence[end_character:]
                        )
                        modified_sentence = (
                            modified_sentence[:start_character]
                            + STARTTAG
                            + modified_sentence[start_character:]
                        )
                        # offset modification shouldn't be needed anymore
                    modified_preds.append(modified_sentence)
                    # Regex's fuzzy matching of edit distance... too slow!
                    # match = regex.search(rf'(?b)(?:{ces["cause"]}){{e}}', sentence)
                d[p["id"]] = {"prediction": modified_preds}
            all_ids = []
            raw_sentences = []
            with open(val_data_grouped, "r") as rf:
                reader = csv.reader(rf, delimiter=",")
                header = next(reader)
                indexcolumn = header.index("index")
                sentencecolumn = header.index("text")
                for row in reader:
                    all_ids.append(row[indexcolumn])
                    raw_sentences.append(row[sentencecolumn])
            r = []
            for index, (id, raw_sentence) in enumerate(zip(all_ids, raw_sentences)):
                if not id in d:
                    r.append({"index": index, "prediction": raw_sentence})
                else:
                    r.append({"index": index, "prediction": d[id]["prediction"]})
            if (
                self.config.get("dump_validation_metadata", False)
                and self.global_rank == 0
            ):
                return r, d
            return r

        if self.config.get("dump_validation_metadata", False) and self.global_rank == 0:
            processed_preds, processed_preds_for_dump = postprocess_predictions(
                dist_predictions, self.config["val_data_grouped"]
            )
            dump = {
                "generation_outputs": dist_predictions,
                "processed_prediction": processed_preds_for_dump,
            }
            with open(
                f"prediction_dump_{os.path.basename(self.config['model_path'])}.pkl",
                "wb",
            ) as wf:
                pickle.dump(dump, wf)
            logger.info("Predicted contents dumped!!")
        else:
            processed_preds = postprocess_predictions(
                dist_predictions, self.config["val_data_grouped"]
            )

        if self.global_rank == 0:
            mkdir(
                os.path.join(self.config["output_predictions_dir"], "input_dir", "res")
            )
            mkdir(
                os.path.join(self.config["output_predictions_dir"], "input_dir", "ref")
            )
            mkdir(os.path.join(self.config["output_predictions_dir"], "output_dir"))
            shutil.copyfile(
                self.config["val_data_grouped"],
                os.path.join(
                    self.config["output_predictions_dir"],
                    "input_dir",
                    "ref",
                    "truth.csv",
                ),
            )

            with jsonlines.open(
                os.path.join(
                    self.config["output_predictions_dir"],
                    "input_dir",
                    "res",
                    "submission.jsonl",
                ),
                "w",
            ) as wf:
                wf.write_all(processed_preds)

            """
            Run official evaluation code, and get performances            
            """

            outputs = run_main_evalf1(
                os.path.join(self.config["output_predictions_dir"], "input_dir"),
                os.path.join(self.config["output_predictions_dir"], "output_dir"),
            )
            for k, v in outputs.items():
                for k2, v2 in v.items():
                    outputs[k][k2] = float(outputs[k][k2])
            logger.debug(json.dumps(outputs, indent=4, sort_keys=True))
            logger.info(f"Validatiom F1 (Overall): {outputs['Overall']['f1']:.5f}\n")

            if flag:
                model.train()
            return outputs["Overall"]["f1"]
        return 0

    def generate_effect_then_cause(
        self,
        cause_prefix,
        effect_prefix,
        history_inputs,
        k,
        max_length,
        metadata,
        model,
        pp,
        src,
        src_mask,
    ):
        if k > 0:
            # prepare effect input
            sentence_seq = [
                CSEDataset.assemble_input_for_effect(m["text"], self.tokenizer)
                for m in metadata
            ]
            inp_effect = []
            for sentence_seq_perex, history_inputs_perex in zip(
                sentence_seq, history_inputs
            ):
                inp_effect_perex = sentence_seq_perex + history_inputs_perex
                if len(inp_effect_perex) > max_length:
                    inp_effect_perex = inp_effect_perex[:max_length]
                inp_effect.append(inp_effect_perex)
            predicted_effects_tokens, predicted_effects = self.get_pred_for_prefix(
                effect_prefix,
                model,
                pp(inp_effect),
                pp([[1] * len(ie) for ie in inp_effect]),
            )
        else:
            predicted_effects_tokens, predicted_effects = self.get_pred_for_prefix(
                effect_prefix, model, src, src_mask
            )

        # prepare cause input
        inp_cause = []
        for m, predicted_effect, history_inputs_perex in zip(
            metadata, predicted_effects, history_inputs
        ):
            sentence_seq = CSEDataset.assemble_input_for_cause(
                m["text"], self.tokenizer, effect_text=predicted_effect
            )
            inp_cause_per_ex = sentence_seq + history_inputs_perex
            if len(inp_cause_per_ex) > max_length:
                inp_cause_per_ex = inp_cause_per_ex[:max_length]
            inp_cause.append(inp_cause_per_ex)
        predicted_causes_tokens, predicted_causes = self.get_pred_for_prefix(
            cause_prefix, model, pp(inp_cause), pp([[1] * len(ic) for ic in inp_cause])
        )
        return (
            predicted_causes,
            predicted_causes_tokens,
            predicted_effects,
            predicted_effects_tokens,
        )

    def generate_cause_then_effect(
        self,
        cause_prefix,
        effect_prefix,
        history_inputs,
        k,
        max_length,
        metadata,
        model,
        pp,
        src,
        src_mask,
    ):
        if k > 0:
            # prepare cause input
            sentence_seq = [
                CSEDataset.assemble_input_for_cause(m["text"], self.tokenizer)
                for m in metadata
            ]
            inp_cause = []
            for sentence_seq_perex, history_inputs_perex in zip(
                sentence_seq, history_inputs
            ):
                inp_cause_perex = sentence_seq_perex + history_inputs_perex
                if len(inp_cause_perex) > max_length:
                    inp_cause_perex = inp_cause_perex[:max_length]
                inp_cause.append(inp_cause_perex)
            predicted_causes_tokens, predicted_causes = self.get_pred_for_prefix(
                cause_prefix,
                model,
                pp(inp_cause),
                pp([[1] * len(ic) for ic in inp_cause]),
            )
        else:
            predicted_causes_tokens, predicted_causes = self.get_pred_for_prefix(
                cause_prefix, model, src, src_mask
            )
        # prepare effect input
        inp_effect = []
        for m, predicted_cause, history_inputs_perex in zip(
            metadata, predicted_causes, history_inputs
        ):
            sentence_seq = CSEDataset.assemble_input_for_effect(
                m["text"], self.tokenizer, cause_text=predicted_cause
            )
            inp_effect_per_ex = sentence_seq + history_inputs_perex
            if len(inp_effect_per_ex) > max_length:
                inp_effect_per_ex = inp_effect_per_ex[:max_length]
            inp_effect.append(inp_effect_per_ex)
        predicted_effects_tokens, predicted_effects = self.get_pred_for_prefix(
            effect_prefix,
            model,
            pp(inp_effect),
            pp([[1] * len(ie) for ie in inp_effect]),
        )
        return (
            predicted_causes,
            predicted_causes_tokens,
            predicted_effects,
            predicted_effects_tokens,
        )

    @torch.no_grad()
    def predict(self):
        config = self.config

        logger.debug(json.dumps(config, indent=4, sort_keys=True))
        if self.distributed:
            distributed_settings = {
                "rank": self.global_rank,
                "world_size": self.world_size,
            }
        else:
            distributed_settings = None

        test = CSEDataset(
            config["test_data"],
            tokenizer=self.tokenizer,
            transformer=config["transformer_type"],
            cache_dir=self.config["data_cache_dir"],
            max_len=self.config.get("max_input_length", None),
            shuffle=False,
            is_training=False,
            grouped=True,
            reversed=self.config.get("ablation", "") == "reverse_cause_effect_genorder",
            distributed_settings=distributed_settings,
        )

        logger.info(f"Test data examples:{len(test)}")

        data_iter = DataLoader(
            test,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=CSEDataset.create_collate_fn(
                pad_t=self.tokenizer.pad_token_id, validate_F1=True
            ),
        )
        logger.info("Loading model...")
        model = self.init_model()

        tokenizer_kwargs = {"add_special_tokens": False, "truncation": False}
        pp = lambda x: torch.LongTensor(x).to(self.torch_device)
        cause_prefix = pp(
            [self.tokenizer.pad_token_id]
            + self.tokenizer.encode("cause: ", **tokenizer_kwargs)
        )
        effect_prefix = pp(
            [self.tokenizer.pad_token_id]
            + self.tokenizer.encode("effect: ", **tokenizer_kwargs)
        )
        signal_prefix = pp(
            [self.tokenizer.pad_token_id]
            + self.tokenizer.encode("signal: ", **tokenizer_kwargs)
        )
        """
        :param model:
        :param data_iter:
        :param optimizer_dict:
        :return:
        """
        flag = model.training
        model = model.eval()

        # Remove gradients, to save memory
        for param in model.parameters():
            param.grad = None

        # Calculate total number of updates per epoch
        total = ceil(len(data_iter.dataset) / data_iter.batch_size)

        # For progressive  training loss  reporting
        predictions = []

        it = tqdm(data_iter, total=total)
        # it = data_iter
        iteration = 0

        def pad_sequences(sequences):
            sequence_lengths = [len(seq) for seq in sequences]
            max_length = max(sequence_lengths)
            padded_sequences = [
                seq + [self.tokenizer.pad_token_id] * (max_length - len(seq))
                for seq in sequences
            ]
            return padded_sequences

        pp = lambda x: torch.LongTensor(pad_sequences(x)).to(self.torch_device)
        max_length = data_iter.dataset.max_len

        start_time = None
        for src, src_mask, target, target_mask, metadata in it:
            if start_time is None:
                start_time = time.time()
            iteration += 1
            src, src_mask = src.to(self.torch_device), src_mask.to(self.torch_device)
            history = [[] for _ in src]
            for k in range(self.config["max_prediction_iterations"]):
                # prepare history for k-th round
                history_inputs = [
                    CSEDataset.assemble_history_inputs(h, self.tokenizer)
                    for h in history
                ]
                if self.config.get("ablation", "") == "reverse_cause_effect_genorder":
                    (
                        predicted_causes,
                        predicted_causes_tokens,
                        predicted_effects,
                        predicted_effects_tokens,
                    ) = self.generate_effect_then_cause(
                        cause_prefix,
                        effect_prefix,
                        history_inputs,
                        k,
                        max_length,
                        metadata,
                        model,
                        pp,
                        src,
                        src_mask,
                    )
                else:
                    (
                        predicted_causes,
                        predicted_causes_tokens,
                        predicted_effects,
                        predicted_effects_tokens,
                    ) = self.generate_cause_then_effect(
                        cause_prefix,
                        effect_prefix,
                        history_inputs,
                        k,
                        max_length,
                        metadata,
                        model,
                        pp,
                        src,
                        src_mask,
                    )
                # prepare signal input
                inp_signal = []
                for m, predicted_cause, predicted_effect, history_inputs_perex in zip(
                    metadata, predicted_causes, predicted_effects, history_inputs
                ):
                    sentence_seq = CSEDataset.assemble_input_for_signal(
                        predicted_causes, predicted_effect, m["text"], self.tokenizer
                    )
                    inp_signal_per_ex = sentence_seq + history_inputs_perex
                    if len(inp_signal_per_ex) > max_length:
                        inp_signal_per_ex = inp_signal_per_ex[:max_length]
                    inp_signal.append(inp_signal_per_ex)

                predicted_signals_tokens, predicted_signals = self.get_pred_for_prefix(
                    signal_prefix,
                    model,
                    pp(inp_signal),
                    pp([[1] * len(ie) for ie in inp_signal]),
                )
                for i in range(len(history)):
                    history[i].append(
                        {
                            "ces_triplet": {
                                "cause": predicted_causes[i],
                                "cause_ids": predicted_causes_tokens[i],
                                "effect": predicted_effects[i],
                                "effect_ids": predicted_effects_tokens[i],
                                "signal": predicted_signals[i],
                                "signal_ids": predicted_signals_tokens[i],
                            }
                        }
                    )
            # logger.info(json.dumps(history, indent=4, sort_keys=True))
            # logger.info("**" * 20)
            for m, h in zip(metadata, history):
                pred = {"sentence": m["text"], "id": m["index"], "predictions": h}
                if self.config.get("dump_validation_metadata", False):
                    pred["metadata"] = m
                    pred["sentence_ids"] = self.tokenizer.encode_plus(
                        m["text"], return_offsets_mapping=True, add_special_tokens=False
                    )
                predictions.append(pred)
        end_time = time.time()
        logger.info(f"Total inference run time: {end_time - start_time:.2f} seconds")
        if self.world_size == "NODDP":
            dist_predictions = predictions
        else:
            dist_predictions = cat_lists(
                share_list(
                    predictions, rank=self.global_rank, world_size=self.world_size
                )
            )

        def postprocess_predictions(dist_predictions, data_grouped):
            d = dict()
            for p in dist_predictions:
                sentence = p["sentence"]
                tokenized = self.tokenizer.encode_plus(
                    sentence, return_offsets_mapping=True, add_special_tokens=False
                )
                # tokenized_offsets = tokenized.offset_mapping
                # tokenized_ids = tokenized.input_ids
                modified_preds = []
                for pred in p["predictions"]:
                    ces = pred["ces_triplet"]

                    cause_span, cause_score = get_bestF1_span(
                        tokenized.input_ids, ces["cause_ids"]
                    )
                    effect_span, effect_score = get_bestF1_span(
                        tokenized.input_ids, ces["effect_ids"]
                    )

                    if ces["signal"] == "empty":
                        signal_span, signal_score = "", 0.0
                    else:
                        signal_span, signal_score = get_bestF1_span(
                            tokenized.input_ids, ces["signal_ids"]
                        )

                    offsets = np.array([[s, e] for s, e in tokenized.offset_mapping])
                    original_words = np.array([sentence[s:e] for s, e in offsets])

                    start_character = end_character = 99999
                    STARTTAG = "<ARG0>"
                    ENDTAG = "</ARG0>"
                    if not cause_score < 0.1:
                        start_character = tokenized.offset_mapping[cause_span[0]][0]
                        end_character = tokenized.offset_mapping[cause_span[1]][1]

                        # modify sentence
                        modified_sentence = (
                            sentence[:end_character] + ENDTAG + sentence[end_character:]
                        )
                        modified_sentence = (
                            modified_sentence[:start_character]
                            + STARTTAG
                            + modified_sentence[start_character:]
                        )
                        # modify offsets
                        offsets[:, 0][offsets[:, 0] > end_character] += len(ENDTAG)
                        offsets[:, 1][offsets[:, 1] > end_character] += len(ENDTAG)
                        offsets[:, 0][offsets[:, 0] >= start_character] += len(STARTTAG)
                        offsets[:, 1][offsets[:, 1] > start_character] += len(STARTTAG)
                        modified_offset_words = np.array(
                            [
                                modified_sentence[s:e]
                                .replace("</ARG0>", "")
                                .replace("<ARG0>", "")
                                for s, e in offsets
                            ]
                        )
                        assert (original_words == modified_offset_words).all()
                    else:
                        modified_sentence = sentence

                    if not effect_score < 0.1:
                        STARTTAG = "<ARG1>"
                        ENDTAG = "</ARG1>"

                        start_character = offsets[effect_span[0]][0]
                        end_character = offsets[effect_span[1]][1]

                        # modify sentence
                        modified_sentence = (
                            modified_sentence[:end_character]
                            + ENDTAG
                            + modified_sentence[end_character:]
                        )
                        modified_sentence = (
                            modified_sentence[:start_character]
                            + STARTTAG
                            + modified_sentence[start_character:]
                        )
                        # modify offsets
                        offsets[:, 0][offsets[:, 0] > end_character] += len(ENDTAG)
                        offsets[:, 1][offsets[:, 1] > end_character] += len(ENDTAG)
                        offsets[:, 0][offsets[:, 0] >= start_character] += len(STARTTAG)
                        offsets[:, 1][offsets[:, 1] > start_character] += len(STARTTAG)
                        modified_offset_words = np.array(
                            [
                                modified_sentence[s:e]
                                .replace("</ARG0>", "")
                                .replace("<ARG0>", "")
                                .replace("</ARG1>", "")
                                .replace("<ARG1>", "")
                                for s, e in offsets
                            ]
                        )
                        assert (original_words == modified_offset_words).all()
                    else:
                        start_character = end_character = 99999

                    if not signal_score < 0.1:
                        STARTTAG = "<SIG0>"
                        ENDTAG = "</SIG0>"
                        start_character = offsets[signal_span[0]][0]
                        end_character = offsets[signal_span[1]][1]

                        # modify sentence
                        modified_sentence = (
                            modified_sentence[:end_character]
                            + ENDTAG
                            + modified_sentence[end_character:]
                        )
                        modified_sentence = (
                            modified_sentence[:start_character]
                            + STARTTAG
                            + modified_sentence[start_character:]
                        )
                        # offset modification shouldn't be needed anymore
                    modified_preds.append(modified_sentence)
                    # Regex's fuzzy matching of edit distance... too slow!
                    # match = regex.search(rf'(?b)(?:{ces["cause"]}){{e}}', sentence)
                d[p["id"]] = {"prediction": modified_preds}
            all_ids = []
            raw_sentences = []
            with open(data_grouped, "r") as rf:
                reader = csv.reader(rf, delimiter=",")
                header = next(reader)
                indexcolumn = header.index("index")
                sentencecolumn = header.index("text")
                for row in reader:
                    all_ids.append(row[indexcolumn])
                    raw_sentences.append(row[sentencecolumn])
            r = []
            for index, (id, raw_sentence) in enumerate(zip(all_ids, raw_sentences)):
                if id not in d:
                    raise ValueError(
                        f"Missing prediction id {id} in prediction dictionary!"
                    )
                r.append({"index": index, "prediction": d[id]["prediction"]})
            return r

        start_time_pp = time.time()
        processed_preds = postprocess_predictions(
            dist_predictions, self.config["test_data"]
        )
        logger.info(
            f"Time spent with postprocessing: {time.time() - start_time_pp} seconds!"
        )

        if self.global_rank == 0:
            mkdir(
                os.path.join(self.config["output_predictions_dir"], "input_dir", "res")
            )
            mkdir(
                os.path.join(self.config["output_predictions_dir"], "input_dir", "ref")
            )
            mkdir(os.path.join(self.config["output_predictions_dir"], "output_dir"))
            prediction_file = os.path.join(
                self.config["output_predictions_dir"],
                "input_dir",
                "res",
                "submission.jsonl",
            )
            with jsonlines.open(prediction_file, "w") as wf:
                wf.write_all(processed_preds)
            logger.info(
                f"{len(processed_preds)} x {self.config['max_prediction_iterations']} predictions written "
                f"into prediction file"
                f"\n{prediction_file}"
            )
