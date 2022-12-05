# IDIAPERS @ CASE22 - SUBTASK 2

<p align="center">
    <a href="https://arxiv.org/abs/2209.03891">
        <img alt="Black" src="https://img.shields.io/badge/arXiv-2209.03891-b31b1b.svg">
    </a>
    <a href="https://github.com/psf/black">
        <img alt="Black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>


## Introduction

This repository contains official code for shared task 3 of [The 5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE @ EMNLP 2022) ](https://emw.ku.edu.tr/case-2022/). Specifically, for subtask-2. 


**In case of questions**, address correspondence to:  
`Martin Fajcik <martin.fajcik@vut.cz>`.

## Table of Contents
- [Downloading Data](#downloading-data)
- [Preparing Environment](#preparing-environment)
- [Scripts](#scripts)
  * [Training](#training)
  * [Inference](#inference)
  * [Hyperparameter Optimization](#hyperparameter-optimization)

## Downloading Data

Go to the project root directory and run `subtask2/download_data_st2.sh`. This will download dataset and our best
checkpoints.

## Preparing Environment

Using `python 3.9.5`, install subtask2 requirements

```bash
python -m pip install -r subtask2/st2_requirements.txt
```

Before running any script, make sure you have `en_US` locale set and `PYTHONPATH` in repository root folder.

```bash
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export PYTHONPATH=$(pwd) #assuming you are in root repository folder
```

## Scripts

Upon entering `subtask2/scripts` directory, you will find all the scripts we used to gather results reported in the
paper.

```python
├── ablations  ## for training ablation models ##
│   ├── __init__.py
│   ├── run_t5_nohistory.py  # trains model ablated as T5-NoHistory
│   └── run_t5_reversedCE.py  # trains model ablated as T5-ECS
├── inference  ## inference on all model types ##
│   ├── __init__.py
│   ├── eval_t5_LARGE.py  # evaluates T5-CES Large checkpoint
│   ├── eval_t5_nohistory.py  # evaluates T5-NoHistory checkpoint
│   ├── eval_t5.py  # evaluates T5-CES checkpoint
│   ├── eval_t5_reversed.py  # evaluates T5-ECS checkpoint
│   ├── predict_t5_LARGE.py  # predicts labels for T5-CES Large checkpoint
│   ├── predict_t5.py  # predicts labels for T5-CES checkpoint
│   └── predict_t5_reversed.py  # predicts labels for T5-ECS checkpoint
├── run_t5_large.py  # trains T5-CES Large model
├── run_t5.py  # trains T5-CES model
└── tune_params_t5base.py  # tunes hyperparameters via HyperOpt

2 directories, 
13 files
```

In all cases, you can modify hyperparameters directly within hardcoded script's `config` dictionary. Each script can be
executed with python.

```bash
python $PATHTOYOURREPO/subtask2/scripts/$SELECTEDSCRIPT.py
```

---
### Training

* All scripts prefixed with `run_` will start the training.
* Large models are trained with `"gradient_checkpointing"` flag set to True, to save memory.
* To have the same loss averaging as reported in the paper (*first average over example, then over minibatch*),
  keep `"batch_size"` at 1.
* All scripts run training in torch DDP mode, utilizing all visible Nvidia GPUs on the machine. Multi-node training is
  not supported.
* All scripts will train 10 models by default. You can change this by modifying `RUNS` variable in the run script.
* All training runs are initially evaluated only by cross-entropy. Only when cross-entropy is lesser than value of `"avoid_f1_computation_when_loss_gt"`, F1 score gets computed. This speeds up training by avoiding slow initial F1 evaluations.
### Inference

* All scripts prefixed with `eval_t5_` will only run evaluation on validation set, as during the training.
    * Keeping `"eval_signal_accuracies"` setting will additionaly evaluate presence of signal span classifier
    * Setting `"dump_validation_metadata"`will dump both **generation_outputs** (generated from T5) and **
      postprocessed_predictions** (matched input parts corresponding to generated outputs)
      into `prediction_dump_<ckptname>.pkl` file for additional analysis.
* All scripts prefixed with `predict_t5_` will infer prediction on test set. Outputs will be written to location
  specified in `"output_predictions_dir"` key, inside nested `submission.jsonl` file. This file can be then submitted to
  CodaLab, or evaluated using official script within `subtask2/evaluation/eval.py` file (if you have labels).
* Inspect `subtask2/dataset/get_best_f1.py` if you are interested in our F1 matching technique (described in the paper).

### Hyperparameter Optimization
Inspect the `subtask2/scripts/tune_params_t5base.py` script. To run the parameter optimization you will need to:
* Run mongoDB on accessible server.
* Run main process, which starts giving jobs, and estimating best hyperparameters via hyperopt.
* Run worker process on every server with GPU, you would like to use for hyperparameter optimization.

To do this:
* Install and **start MongoDB** to some server with accessible IP. The hardcoded DB port is `1234`. You can start mongo in local folder with command like:
```bash
user@mymongoserver:~$ mongod --port 1234 --bind_ip_all --dbpath ./my_local_folder
```
* Inside `tune_params_t5base` script, set `SERVER` variable to domain name of your server (e.g. `serverxyz.myorg.com`). You can change the port number here, if necessary.
* Set path of your repository by modifying `PATH` variable inside the `obj` function. Your repository may lie in a different path for every machine, and this variable makes sure the optimization script starts in correct directory.
* Now you can **start main hyperopt process** by running `subtask2/scripts/tune_params_t5base.py`.
* **Start workers** on machines with GPUs. First login to machine with GPU and navigate to (possibly mounted) repository directory. Set your `PYTHONPATH` to the repository root, and configure locale as before. Then you need to install same python environment on these machines. Finally, worker can be started by running
```bash
hyperopt-mongo-worker --mongo=$MONGODBSERVER:1234/ce_t5_db --poll-interval=0.5
```
