# IDIAPERS @ CASE22-TASK 3: Event Causality Identification

<p align="center">
    <a href="https://github.com/idiap/cncsharedtask/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-green.svg">
    </a>
    <a href="https://github.com/idiap/cncsharedtask">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Open%20source-green">
    </a>
    <a href="https://github.com/psf/black">
        <img alt="Black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://arxiv.org/abs/2209.03895">
        <img alt="Black" src="https://img.shields.io/badge/arXiv-2209.03895-b31b1b.svg">
    </a>
    <a href="https://arxiv.org/abs/2209.03891">
        <img alt="Black" src="https://img.shields.io/badge/arXiv-2209.03891-b31b1b.svg">
    </a>
</p>


## Introduction

This repository contains official code for shared task 3 of [The 5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE @ EMNLP 2022) ](https://emw.ku.edu.tr/case-2022/)


The task 3 was comprised of two subtasks. Given a sentence from the news-media:
* Subtask 1: identify whether sentence contains any causal relation,
* Subtask 2: extract all cause-effect-signal triplets capturing causal relations from this sentence.

Causality is a core cognitive concept and appears in many natural language processing (NLP) works that aim to tackle inference and understanding. Generally, a causal relation is a semantic relationship between two arguments known as cause and effect, in which the occurrence of one (cause argument) causes the occurrence of the other (effect argument). The Figure below illustrates some sentences that are marked as <em>Causal</em> and <em>Non-causal</em> respectively.


| <img align="center" height=250 src="https://github.com/tanfiona/CausalNewsCorpus/blob/master/imgs/EventCausality_Subtask1_Examples3.png"> | 
|:--:| 
| *Annotated examples from Causal News Corpus. Causes are in pink, Effects in green and Signals in yellow. Note that both Cause and Effect spans must be present within one and the same sentence for us to mark it as <em>Causal</em>. Figure taken from official challenge github.* |


More information can be found at the [official challenge github](https://github.com/tanfiona/CausalNewsCorpus).

## Installation
The installation instructions for each subtask can be found in README located its respective folder (`subtask1` and `subtask2` respectively).



# Citation
If you use our work or code, please cite our works for respective subtask
* Subtask 1: [IDIAPers @ Causal News Corpus 2022: Efficient Causal Relation Identification Through a Prompt-based Few-shot Approach](https://arxiv.org/abs/2209.03895)
* Subtask 2: [IDIAPers @ Causal News Corpus 2022: Extracting Cause-Effect-Signal Triplets via Pre-trained Autoregressive Language Model](https://arxiv.org/abs/2209.03891)

Bibtex citations:
```bibtex
@inproceedings{idiap_case22_subtask1,
    title = "{IDIAPers} @ Causal News Corpus 2022: Causal Relation Identification Using a Few-shot and Prompt-based Fine-tuning of Language Models",
    author = "Burdisso, Sergio and Zuluaga-Gomez, Juan and Fajcik, Martin and Villatoro-Tello, Esaú and Singh, Muskaan and Motlicek, Petr and Smrz, Pavel",
    booktitle = "The 5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE @ EMNLP 2022)",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```


```bibtex
@inproceedings{idiap_case22_subtask2,
    title = "IDIAPers @ Causal News Corpus 2022: Extracting Cause-Effect-Signal Triplets via Pre-trained Autoregressive Language Model",
    author = "Fajcik, Martin and Singh, Muskaan and Zuluaga-Gomez, Juan and Villatoro-Tello, Esaú and Burdisso, Sergio and Motlicek, Petr and Smrz, Pavel",
    booktitle = "The 5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE @ EMNLP 2022)",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

# Getting Help
If you need help, don't hesitate to create an issue at GitHub, or write to corresponding author.