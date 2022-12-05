# IDIAPERS @ CASE22 - SUBTASK 1

<p align="center">
    <a href="https://arxiv.org/abs/2209.03895">
        <img alt="Black" src="https://img.shields.io/badge/arXiv-2209.03895-b31b1b.svg">
    </a>
    <a href="https://github.com/psf/black">
        <img alt="Black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>


## Introduction

This repository contains official code for shared task 3 of [The 5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE @ EMNLP 2022) ](https://emw.ku.edu.tr/case-2022/). Specifically, for subtask-1. 


**In case of questions**, address correspondence to:  
`Sergio Burdisso <sergio.burdisso@idiap.ch>`.

## Table of Contents
- [Preparing The Data](#preparing-the-data)
- [Preparing Environment](#preparing-environment)
- [Scripts](#scripts)
  * [Training](#training)
  * [Inference](#inference)
  * [Hyperparameter Optimization](#hyperparameter-optimization)

## Preparing The Data

Go to the project root directory and run `bash data/scripts/gen_kfolds_all.sh`. This will prepare the 5-folds used in the original CASE paper and also suggested by the datasets authors.

## Preparing Environment

Using `python 3.9.5`, install subtask1 requirements

```bash
python -m pip install -r subtask1/baseline_lms/requirements.txt
python -m pip install -r subtask1/prompt_based/requirements.txt
```

Before running any script, make sure you have `en_US` locale set and `PYTHONPATH` in repository root folder.

```bash
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export PYTHONPATH=$(pwd) #assuming you are in root repository folder
```

## Scripts

Upon entering `subtask1/` directory, you will find all the scripts we used to gather results reported in the paper. The results are split in two. 1) Baseline experiments, where we simply fine-tuned LMs in the sequence classification task. 2) The prompt-based experiments.


**Baseline experiments:**

```python
.
├── baseline_lms
│   ├── baselines
│   │   ├── ensemble    # scripts for training n models and running ensemble
│   │   │   ├── ensemble_debertav3.sh
│   │   │   └── get_ensemble_f1.py  # get the best ensemble, given n models
│   │   ├── run_all_models_only_cnc_data.sh  # run fine-tuning of several pre-trained LMs
│   │   └── train_baseline_bert.sh # train a baseline with BERT-base-cased
│   ├── evaluations # script for running the evaluation 
│   │   ├── eval_model.sh
│   │   └── run_inference.py
│   ├── reformat_output.py
│   ├── requeriments.txt
│   ├── run_case.py # main python script for training the baselines (HuggingFace styled)
│   └── utils   # scripts to parse input from CLI
│       ├── parse_options.sh
│       └── queue.pl
├── data    # folder with the data for replicate our results
│   ├── dev_subtask1.csv
│   ├── dev_subtask1_text.csv
│   ├── README.md
│   ├── scripts
│   │   ├── gen_kfolds_all.sh   # generate the folds for k-fold CV
│   │   ├── gen_kfolds.py
│   │   └── utils.py
│   ├── test_subtask1_text.csv
│   └── train_subtask1.csv
└── README.md

```

**Prompt-based experiments:**

```python
.
├── prompt_based
│   ├── gather_best_results.py
│   ├── get_metric_details.py
│   ├── README.md
│   ├── requirements.txt
│   ├── run_ensemble.py
│   ├── run.py
│   ├── src
│   │   ├── dataset.py
│   │   ├── label_search.py
│   │   ├── loss.py
│   │   ├── models.py
│   │   ├── processors.py
│   │   ├── trainer.py
│   │   └── trainer_utils.py
│   └── tools
│       ├── generate_template.py
│       ├── get_sbert_embedding.py
│       └── sort_template.py
└── README.md
```

# TODO

---
### Training

- Training: ...

### Inference

- Inference:...

