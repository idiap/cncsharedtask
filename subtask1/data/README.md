## Dataset Description

#### Subtask 1 

Columns:
* index [str] : example unique id
* text [str] : example input text
* label [int] : target causal label (1 for Causal, 0 for Not Causal)
* agreement [float] : proportion of annotators supporting the vote
* num_votes [int] : number of expert labels considered 
* sample_set [str] : subset name


## Shared Task
The following files are relevant for the on-going shared task on Codalab.

#### Evaluation Phase (Apr 15, 2022 -- Aug 01, 2022)

###### Subtask 1
* Train: train_subtask1.csv
* Test: dev_subtask1_text.csv

###### Subtask 2
* Train: train_subtask2.csv
* Test: dev_subtask2_text.csv

#### Testing Phase (Aug 01, 2022 -- Aug 15, 2022)

###### Subtask 1
* Train: train_subtask1.csv & dev_subtask1.csv
* Test: test_subtask1_text.csv

###### Subtask 2
* Train: train_subtask2.csv & dev_subtask2.csv
* Test: test_subtask2_text.csv
