#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License

import logging
import os
import pickle
import socket
from random import randint

import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_ddp_envinit(rank, world_size, backend, address=None, port=None):
    address = "localhost" if address is None else address
    port = str(randint(3000, 15000)) if port is None else str(port)
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = port

    logger.info(f"Setup env master address/port {address}:{port}")
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def setup_ddp_tcpinit(rank, world_size, backend, address, port):
    method = f"tcp://{address}:{port}"

    logger.info(f"Setup tcp address/port {address}:{port}")
    # initialize the process group
    dist.init_process_group(
        backend, init_method=method, rank=rank, world_size=world_size
    )


def setup_ddp_fsinit(rank, world_size, backend, path):
    # method = f'file://{path}'
    #
    # logger.info(f"Setup fs ddp at {path}")
    # # initialize the process group
    # dist.init_process_group(backend, init_method=method, rank=rank, world_size=world_size)
    # logger.info(f"Finished setup fs ddp at {path}")
    store = dist.FileStore(path, world_size)
    logger.info(f"Setup fs ddp at {path}")
    dist.init_process_group(backend, store=store, rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def share_list(l, rank, world_size, log=False):
    "Just use pickle"
    assert type(l) == list
    PGID = os.getpgid(os.getpid())
    out_fn = (
        f"__shared_list_H{socket.gethostname()}_PGID{PGID}_R{rank}_W{world_size}.pkl"
    )
    with open(out_fn, "wb") as f:
        pickle.dump(l, f)
    if log:
        logger.debug(
            f"R {dist.get_rank()} Data dumped. Approaching barrier #1 {out_fn}"
        )
    dist.barrier()
    if log:
        logger.debug(f"R {dist.get_rank()} Reading {out_fn}")
    all_files = [
        f"__shared_list_H{socket.gethostname()}_PGID{PGID}_R{r}_W{world_size}.pkl"
        for r in range(world_size)
    ]
    lists = []
    for in_fn in all_files:
        with open(in_fn, "rb") as f:
            lists.append(pickle.load(f))
    if log:
        logger.debug(f"R {dist.get_rank()} Pre-Barrier #2 {out_fn}")
    dist.barrier()
    os.remove(out_fn)
    if log:
        logger.debug(f"R {dist.get_rank()} Deleted {out_fn}")
    return lists
