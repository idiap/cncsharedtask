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
import os
import pickle
import random
import re
import time
from collections import defaultdict
from itertools import permutations
from math import ceil
from random import shuffle
from typing import AnyStr

import torch
import torch.distributed as dist
from jsonlines import jsonlines
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from subtask2.utils.utility import mkdir

logger = logging.getLogger(__name__)


class CSEDataset(IterableDataset):
    def __init__(
        self,
        data_file: AnyStr,
        tokenizer: PreTrainedTokenizer,
        transformer: AnyStr,
        max_len=None,
        is_training=True,
        shuffle=False,
        cache_dir="data/preprocessed",
        grouped=False,
        reversed=False,
        skip_history=False,
        distributed_settings=None,
    ):
        self.cache_dir = cache_dir
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_len = (
            max_len if max_len is not None else self.tokenizer.model_max_length
        )
        self.is_training = is_training
        self.shuffle = shuffle
        self.distributed_settings = distributed_settings
        self.grouped = grouped
        self.reversed = reversed
        self.skip_history = skip_history

        preprocessed_f = self.create_preprocessed_name()
        self.preprocessed_f = preprocessed_f
        if not os.path.exists(preprocessed_f):
            logger.info(
                f"{preprocessed_f} not found!\nCreating new preprocessed file..."
            )
            if distributed_settings is not None:
                if dist.get_rank() == 0:
                    self.preprocess_data(preprocessed_f)
                dist.barrier()  # wait for preprocessing to be finished by 0
            else:
                self.preprocess_data(preprocessed_f)

        self.index_dataset()
        self._total_data = len(self._line_offsets)
        if distributed_settings is not None:
            standard_part_size = ceil(
                self._total_data / distributed_settings["world_size"]
            )
            self._total_data_per_rank = (
                standard_part_size
                if (
                    distributed_settings["rank"]
                    < distributed_settings["world_size"] - 1
                    or is_training
                )
                else self._total_data
                - (distributed_settings["world_size"] - 1) * standard_part_size
            )

    def preprocess_data(self, preprocessed_f):
        s_time = time.time()
        self._preprocess_data()
        logger.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")

    @staticmethod
    def assemble_input_sequence(history, target, tokenizer, max_length, reversed=False):
        """
        Prepare inputs for S2S model, including history preds & current sentence to predict
        """

        # Assemble history + cause/effect/signal as separate samples,
        # to be less prone to generation errors (not generating certain segments...)
        history_inputs = CSEDataset.assemble_history_inputs(history, tokenizer)

        ces = target["ces_triplet"]

        # cause input
        if not reversed:
            sentence_seq = CSEDataset.assemble_input_for_cause(
                target["text"], tokenizer
            )
        else:
            sentence_seq = CSEDataset.assemble_input_for_cause(
                target["text"], tokenizer, effect_text=ces["effect"]
            )
        inp_cause = sentence_seq + history_inputs
        if len(inp_cause) > max_length:
            inp_cause = inp_cause[:max_length]

        # effect input
        if not reversed:
            sentence_seq = CSEDataset.assemble_input_for_effect(
                target["text"], tokenizer, cause_text=ces["cause"]
            )
        else:
            sentence_seq = CSEDataset.assemble_input_for_effect(
                target["text"], tokenizer
            )
        inp_effect = sentence_seq + history_inputs
        if len(inp_effect) > max_length:
            inp_effect = inp_effect[:max_length]

        # signal input
        sentence_seq = CSEDataset.assemble_input_for_signal(
            ces["cause"], ces["effect"], target["text"], tokenizer
        )
        inp_signal = sentence_seq + history_inputs

        if len(inp_signal) > max_length:
            inp_signal = inp_signal[:max_length]
        if reversed:
            return inp_effect, inp_cause, inp_signal

        return inp_cause, inp_effect, inp_signal

    @staticmethod
    def assemble_input_for_cause(sentence, tokenizer, effect_text=None):
        tokenizer_kwargs = {"add_special_tokens": False, "truncation": False}
        if effect_text is not None:
            sentence_seq = tokenizer.encode(
                sentence + f" effect:{effect_text}", **tokenizer_kwargs
            )
        else:
            sentence_seq = tokenizer.encode(sentence, **tokenizer_kwargs)
        return sentence_seq

    @staticmethod
    def assemble_input_for_signal(cause, effect, sentence, tokenizer):
        tokenizer_kwargs = {"add_special_tokens": False, "truncation": False}
        sentence_seq = tokenizer.encode(
            sentence + f" cause:{cause} effect:{effect}", **tokenizer_kwargs
        )
        return sentence_seq

    @staticmethod
    def assemble_input_for_effect(sentence, tokenizer, cause_text=None):
        tokenizer_kwargs = {"add_special_tokens": False, "truncation": False}
        if cause_text is not None:
            sentence_seq = tokenizer.encode(
                sentence + f" cause:{cause_text}", **tokenizer_kwargs
            )
        else:
            sentence_seq = tokenizer.encode(sentence, **tokenizer_kwargs)
        return sentence_seq

    @staticmethod
    def assemble_history_inputs(history, tokenizer):
        tokenizer_kwargs = {"add_special_tokens": False, "truncation": False}
        encoded_history_prefix = tokenizer.encode("history: ", **tokenizer_kwargs)
        history_inputs = encoded_history_prefix
        for h in history:
            hces = h["ces_triplet"]

            hseq = (
                tokenizer.encode("cause: ", **tokenizer_kwargs)
                + tokenizer.encode(hces["cause"], **tokenizer_kwargs)
                + tokenizer.encode("effect: ", **tokenizer_kwargs)
                + tokenizer.encode(hces["effect"], **tokenizer_kwargs)
                + tokenizer.encode("signal: ", **tokenizer_kwargs)
                + tokenizer.encode(hces["signal"], **tokenizer_kwargs)
            )
            history_inputs += hseq
        return history_inputs

    @staticmethod
    def assemble_target_sequence(target, tokenizer, max_length, reversed=True):
        """
        Prepare target to predict
        """
        # Assemble history + cause/effect/signal as separate samples,
        # to be less prone to generation errors (not generating certain segments...)
        tokenizer_kwargs = {"add_special_tokens": False, "truncation": False}
        ces = target["ces_triplet"]
        target_seq_cause = tokenizer.encode(
            "cause: ", **tokenizer_kwargs
        ) + tokenizer.encode(ces["cause"], **tokenizer_kwargs)
        target_seq_effect = tokenizer.encode(
            "effect: ", **tokenizer_kwargs
        ) + tokenizer.encode(ces["effect"], **tokenizer_kwargs)
        target_seq_signal = tokenizer.encode(
            "signal: ", **tokenizer_kwargs
        ) + tokenizer.encode(ces["signal"], **tokenizer_kwargs)

        # t5 starts generation with pad token
        target_seq_cause = (
            [tokenizer.pad_token_id] + target_seq_cause + [tokenizer.eos_token_id]
        )
        if len(target) > max_length:
            target_seq_cause = target_seq_cause[: max_length - 1] + [
                tokenizer.eos_token_id
            ]

        target_seq_effect = (
            [tokenizer.pad_token_id] + target_seq_effect + [tokenizer.eos_token_id]
        )
        if len(target) > max_length:
            target_seq_effect = target_seq_effect[: max_length - 1] + [
                tokenizer.eos_token_id
            ]

        target_seq_signal = (
            [tokenizer.pad_token_id] + target_seq_signal + [tokenizer.eos_token_id]
        )
        if len(target) > max_length:
            target_seq_signal = target_seq_signal[: max_length - 1] + [
                tokenizer.eos_token_id
            ]

        if reversed:
            return target_seq_effect, target_seq_cause, target_seq_signal
        return target_seq_cause, target_seq_effect, target_seq_signal

    @staticmethod
    def process_example(e, skip_history=False, **kwargs):
        all_paths = []
        # create path for every permutation
        for ep in permutations(e):
            ep = list(ep)
            encoded_path = []
            for i in range(len(ep)):
                if skip_history:
                    input_sequences = CSEDataset.assemble_input_sequence(
                        history=[], target=ep[i], **kwargs
                    )
                else:
                    input_sequences = CSEDataset.assemble_input_sequence(
                        history=ep[:i], target=ep[i], **kwargs
                    )
                target_sequences = CSEDataset.assemble_target_sequence(
                    target=ep[i], **kwargs
                )
                encoded_path.append(
                    {
                        "input_sequences": input_sequences,
                        "target_sequences": target_sequences,
                        "metadata": ep[: i + 1] if not skip_history else [ep[i]],
                    }
                )

            all_paths.append(encoded_path)
        return all_paths

    def _preprocess_data(self):
        # Load dataset
        with open(self.data_file) as fhandle:
            csv_reader = csv.reader(fhandle, delimiter=",")
            HEADER = next(csv_reader)
            if self.grouped:
                dataset = []
            else:
                dataset_dict = defaultdict(lambda: [])
            for row in csv_reader:
                metadata = dict(zip(HEADER, row))
                if "text_w_pairs" in metadata:
                    cause, effect, signal = self.parse_causal_triplet(
                        metadata["text_w_pairs"]
                    )
                else:
                    cause, effect, signal = "", "", ""
                metadata["ces_triplet"] = {
                    "cause": cause,
                    "effect": effect,
                    "signal": signal,
                }
                if self.grouped:
                    dataset.append([metadata])
                else:
                    dataset_dict[metadata["text"]].append(metadata)

        if not self.grouped:
            # 1 example = 1 unique sentence with (possibly) multiple annotations
            dataset = list(dataset_dict.values())
        mkdir(os.path.dirname(self.preprocessed_f))
        # Preprocess dataset
        with jsonlines.open(self.preprocessed_f, "w") as wf:
            for e in dataset:
                processed_example = CSEDataset.process_example(
                    e,
                    tokenizer=self.tokenizer,
                    max_length=self.max_len,
                    skip_history=self.skip_history,
                    reversed=self.reversed,
                )
                wf.write(processed_example)

    arg0_regex = re.compile(r"(<ARG0>)(.*)(<\/ARG0>)")
    arg1_regex = re.compile(r"(<ARG1>)(.*)(<\/ARG1>)")
    sig0_regex = re.compile(r"(<SIG0>)(.*)(<\/SIG0>)")

    @staticmethod
    def parse_causal_triplet(s):
        def replace_labels(s):
            s = s.replace("<ARG0>", "")
            s = s.replace("</ARG0>", "")
            s = s.replace("<ARG1>", "")
            s = s.replace("</ARG1>", "")
            s = s.replace("<SIG0>", "")
            s = s.replace("</SIG0>", "")
            return s

        if s == "":
            return "", "", ""
        match_arg0 = CSEDataset.arg0_regex.search(s.strip())
        match_arg1 = CSEDataset.arg1_regex.search(s.strip())
        match_sig0 = CSEDataset.sig0_regex.search(s.strip())
        if match_arg0:
            arg0 = match_arg0.group()
            arg0 = replace_labels(arg0)
        if match_arg1:
            arg1 = match_arg1.group()
            arg1 = replace_labels(arg1)
        if match_sig0:
            sig0 = match_sig0.group()
            sig0 = replace_labels(sig0)
        else:
            sig0 = "empty"
        return arg0, arg1, sig0

    def __len__(self):
        return (
            self._total_data if not dist.is_initialized() else self._total_data_per_rank
        )

    def get_example(self, n: int) -> str:
        """
        Get n-th line from dataset file.
        :param n: Number of line you want to read.
        :type n: int
        :return: the line
        :rtype: str
        """
        if self.preprocessed_f_handle.closed:
            self.preprocessed_f_handle = open(self.preprocessed_f)

        self.preprocessed_f_handle.seek(self._line_offsets[n])
        return json.loads(self.preprocessed_f_handle.readline().strip())

    def index_dataset(self):
        """
        Makes index of dataset. Which means that it finds offsets of the samples lines.
        """

        lo_cache = self.preprocessed_f + "locache.pkl"
        # Line offsets are pre-cached, if possible
        if os.path.exists(lo_cache):
            logger.info(f"Using cached line offsets from {lo_cache}")
            with open(lo_cache, "rb") as f:
                self._line_offsets = pickle.load(f)
        else:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    logger.info(f"Getting lines offsets in {self.preprocessed_f}")
                    self._index_dataset(lo_cache)
                dist.barrier()  # wait for precaching to be finished by 0
                if dist.get_rank() > 0:
                    logger.info(f"Using cached line offsets from {lo_cache}")
                    with open(lo_cache, "rb") as f:
                        self._line_offsets = pickle.load(f)
            else:
                logger.info(f"Getting lines offsets in {self.preprocessed_f}")
                self._index_dataset(lo_cache)

    def _index_dataset(self, lo_cache):
        self._line_offsets = [0]
        with open(self.preprocessed_f, "rb") as f:
            while f.readline():
                self._line_offsets.append(f.tell())
        del self._line_offsets[-1]
        # cache file index
        with open(lo_cache, "wb") as f:
            pickle.dump(self._line_offsets, f)

    def __iter__(self):
        self.preprocessed_f_handle = open(self.preprocessed_f)
        self.order = list(range(self._total_data))
        if self.shuffle:
            # logger.info("Shuffling file index...")
            shuffle(self.order)
        if dist.is_initialized():
            distributed_shard_size = ceil(
                self._total_data / self.distributed_settings["world_size"]
            )
            self.shard_order = self.order[
                self.distributed_settings["rank"]
                * distributed_shard_size : (self.distributed_settings["rank"] + 1)
                * distributed_shard_size
            ]
            if len(self.shard_order) < distributed_shard_size and self.is_training:
                logger.info(
                    f"Padding process {os.getpid()} with rank {self.distributed_settings['rank']} with "
                    f"{distributed_shard_size - len(self.shard_order)} samples"
                )
                self.padded = distributed_shard_size - len(self.shard_order)
                self.shard_order += self.order[
                    : distributed_shard_size - len(self.shard_order)
                ]
            self.order = self.shard_order
        self.offset = 0
        return self

    def __next__(self):
        if self.offset >= len(self.order):
            if not self.preprocessed_f_handle.closed:
                self.preprocessed_f_handle.close()
            raise StopIteration
        example = self.get_example(self.order[self.offset])
        self.offset += 1
        return example

    def create_preprocessed_name(self):
        transformer = self.transformer.replace("/", "_")
        maxlen = f"_L{self.max_len}" if self.max_len is not None else ""
        preprocessed_f_noext = (
            os.path.join(self.cache_dir, os.path.basename(self.data_file))
            + f"_CSEextractor_preprocessed_for"
            f"_{transformer}"
            f"{'_grouped' if self.grouped else ''}"
            f"{'_reversed' if self.reversed else ''}"
            f"{'_skip_history' if self.skip_history else ''}"
            f"{maxlen}"
        )
        preprocessed_f = preprocessed_f_noext + ".jsonl"
        return preprocessed_f

    @staticmethod
    def create_collate_fn(pad_t, validate_loss=False, validate_F1=False):
        def pad_sequences(sequences):
            sequence_lengths = [len(seq) for seq in sequences]
            max_length = max(sequence_lengths)
            padded_sequences = [
                seq + [pad_t] * (max_length - len(seq)) for seq in sequences
            ]
            return padded_sequences

        def collate_fn(batch):
            (
                sources_list,
                source_masks_list,
                targets_list,
                target_masks_list,
                metadata_list,
            ) = ([], [], [], [], [])
            for paths in batch:
                if validate_loss:
                    # During validation, the loss is computed for all possible paths (so it is deterministic)
                    for path in paths:
                        for pathidx, e in enumerate(path):
                            assert len(e["input_sequences"]) == len(
                                e["target_sequences"]
                            )
                            for s, t in zip(
                                e["input_sequences"], e["target_sequences"]
                            ):
                                sources_list.append(s)
                                source_masks_list.append([1] * len(s))
                                targets_list.append(t)
                                target_masks_list.append([1] * len(t))
                                e["metadata"][-1]["pathidx"] = pathidx
                                metadata_list.append(e["metadata"])
                elif validate_F1:
                    # In F1 validation, we only need raw sentence for computation
                    e = paths[0][0]
                    s = e["input_sequences"][0]
                    t = [-1]

                    sources_list.append(s)
                    source_masks_list.append([1] * len(s))
                    targets_list.append(t)
                    target_masks_list.append([1] * len(t))
                    metadata_list.append(e["metadata"][0])
                else:  # during training, we sample just 1 path
                    path = random.choice(paths)  # choose random permutation
                    for pathidx, e in enumerate(path):
                        assert len(e["input_sequences"]) == len(e["target_sequences"])
                        for s, t in zip(e["input_sequences"], e["target_sequences"]):
                            sources_list.append(s)
                            source_masks_list.append([1] * len(s))
                            targets_list.append(t)
                            target_masks_list.append([1] * len(t))
                            e["metadata"][-1]["pathidx"] = pathidx
                            metadata_list.append(e["metadata"][0])

            pp = lambda x: torch.LongTensor(pad_sequences(x))
            return (
                pp(sources_list),
                pp(source_masks_list),
                pp(targets_list),
                pp(target_masks_list),
                metadata_list,
            )

        return collate_fn
