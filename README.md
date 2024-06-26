# TEAM – Threat Embracing by Automated Methods

This repository contains the set of methods for the article "TEAM – Threat Embracing by Automated Methods".

The execution of TEAM 2 and TEAM 3 is guided through Jupyter Notebooks.

The notebook [TEAM_2.ipynb](https://github.com/tsumarios/TEAM/blob/main/TEAM_2.ipynb) implements TEAM 2.
The notebook [TEAM_3.ipynb](https://github.com/tsumarios/TEAM/blob/main/TEAM_3.ipynb) implements TEAM 3.

Each of the two notebooks include detailed instructions to guide the analyst throughout the execution of the TEAM methods.

## Utils

The module [embracing_utils.py](https://github.com/tsumarios/TEAM/blob/main/embracing_utils.py) implements the core functionalities for both TEAM 2 and TEAM 3, in particular for the semantic similarity and semantic relations.

The script [ss_scores_by_groups.py](https://github.com/tsumarios/TEAM/blob/main/embracing_utils.py) computes the semantic similarity scores for tuples of cardinality *k* for TEAM 2, where *k* needs to be set in the main function of the script.

## Data

The folder [data](https://github.com/tsumarios/TEAM/blob/main/data/) contains the inputs and outputs for TEAM 2 and TEAM 3, conveniently structured into subfolders.

For storage reasons, [TEAM 2](https://github.com/tsumarios/TEAM/tree/main/data/TEAM%202) only contains the semantic similarity scores computed for tuples with cardinality 3.

[TEAM 3](https://github.com/tsumarios/TEAM/tree/main/data/TEAM%203) contains the results of the first, second and third round, as well as the validation results. In particular, each of the iterations includes semantic relations (synset relations) for embraceable candidates. The validation results compare TEAM 3 with TEAM 1.

## Statistics and Images

The folder [img](https://github.com/tsumarios/TEAM/tree/main/img) provides some plots for both TEAM 2 and TEAM 3 statistics.

### PoC

The notebook [poc.ipynb](https://github.com/tsumarios/TEAM/blob/main/poc.ipynb) provides a Proof-of-Concept for the fundamentals behind the TEAM methods.
