#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License

from transformers import AutoTokenizer, PreTrainedTokenizer


def init_tokenizer(config, tokenizer_class=AutoTokenizer) -> PreTrainedTokenizer:
    """
    Creates tokenizer
    """
    tokenizer = tokenizer_class.from_pretrained(
        config["tokenizer_type"], cache_dir=config["transformers_cache"]
    )

    return tokenizer
