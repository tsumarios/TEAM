# TEAM – Threat Embracing by Automated Methods

This repository contains methods for the article "TEAM – Threat Embracing by Automated Methods."

The purpose of TEAM is to automate threat embracing aiding the threat elicitation process within the [SPADA methodology for threat modelling](https://github.com/tsumarios/Threat-Modelling-Research/tree/main/SPADA).

The repository provides methods for both TEAM 2 and TEAM 3, accessible via Jupyter Notebooks or through an interactive Streamlit web application.

## Table of Contents

- [Usage](#usage)
  - [Prerequisites](#prerequisites)
    - [Prepare the Input Threats CSV](#prepare-the-input-threats-csv)
    - [Obtain the Semantic Similarity Scores CSV](#obtain-the-semantic-similarity-scores-csv)
  - [Choose a TEAM Tool](#choose-a-team-tool)
    - [TEAM Notebooks](#team-notebooks)
    - [TEAM Web Application](#team-web-application)
- [TEAM Core and Utils](#team-core-and-utils)
  - [Embracing Utils](#embracing-utils)
  - [Compute Tuple Similarity Scores](#compute-tuple-similarity-scores)
- [Data](#data)
- [Statistics and Images](#statistics-and-images)
- [PoC](#poc)
- [Contacts](#contacts)

## Usage

Here's a quick guide to get started.

### Prerequisites

First of all, don't forget to install dependencies:

```sh
pip install -r requirements.txt
```

#### Prepare the Input Threats CSV

The Input Threats CSV must have the following columns:

**(ID,Threat,Source of documentation)**

Where:

- ID: Identifies the threat
- Threat: Describes the threat
- Source of documentation: Specifies the document source which the threat was retrieved from.

Example:

*(t_19,Manipulation of information,ENISA)*

#### Obtain the Semantic Similarity Scores CSV

For performance reasons (unless you are using the [TEAM_3.ipynb] notebook), the[TEAM_2.ipynb] notebook and the [TEAM_app.py] web app both require the user to run the ``compute_tuple_similarity_scores.py`` script outside of the tool. This generates similarity scores for each tuple and stores the results as ``./data/{input_threats_filename}_ss_scores_with_cardinality_{k}.csv``.

Open you favourite Terminal and type as follows:

```sh
python3 compute_tuple_similarity_scores.py --k <k> --in_path ./data/input_threats.csv [--previous_k_scores_path ./data/previous_k_semantic_similarity_scores.csv] --out_path ./data/output_similarity_scores.csv
```

Where:

- --k <k>: Specifies the tuple cardinality for similarity scoring.
- --in_path: Path to the Input Threats CSV file.
- --previous_k_scores_path: Path to the precomputed Semantic Similarity Scores CSV related to k-1 (only needed if k > 2).
- --out_path: Path to the output file for the computed similarity scores.

You can choose to compute similarity scores either using the TEAM `compute_tuple_similarity_score` script or adopting different methods. In the latter case, just make sure the Input Threats CSV and Semantic Similarity Scores CSV follow the same structure as expected by TEAM.

The Semantic Similarity Scores CSV must have the following columns:

- *(sentence1,sentence2,score)* for k=2;
- *(max,mean,min,scores,sentence1,sentence2,...,sentencek)* for k>2.

### Choose a TEAM Tool

Once you have both the Input Threat CSV and the Semantic Similarity Score CSV, as a result of the above script, you can proceed by following the steps illustrated in the TEAM notebooks or TEAM web app.

#### TEAM Notebooks

The execution of TEAM 2 and TEAM 3 is guided through the following notebooks:

- [TEAM_2.ipynb](https://github.com/tsumarios/TEAM/blob/main/TEAM_2.ipynb) implements TEAM 2.
- [TEAM_3.ipynb](https://github.com/tsumarios/TEAM/blob/main/TEAM_3.ipynb) implements TEAM 3.

Each of the two notebooks include detailed instructions to guide the analyst throughout the execution of the TEAM methods.

#### TEAM Web Application

The [TEAM_app.py`](https://github.com/tsumarios/TEAM/blob/main/TEAM_app.py) Streamlit web app provides an interactive interface for threat embracing in TEAM. To launch the app, use:

```sh
streamlit run TEAM_app.py
```

While the full guide is available within the app at the top of the page, here is a brief summary:

- *File Uploads*: Upload the CSV of threats and the CSV of semantic similarity scores for each threat tuple.
- *Configuration*: Customise threshold, tuple cardinality (k), and aggregation method to identify and filter embraceable threat candidates. Optionally, assign a custom prefix for threat IDs for organised tracking.
- *Embracing operation*: Select tuples to embrace, specifying a conductor (lead threat) and orchestra (embraced threats). Optionally rename or discard threats as needed.
- *Exports*: Save the results in CSV format for both embraced threats and the updated list of threats.

## TEAM Core and Utils

The following components are the core of TEAM and utils.

### Embracing Utils

The script [embracing_utils.py](https://github.com/tsumarios/TEAM/blob/main/embracing_utils.py) provides utility functions for semantic similarity computation, synset relations analysis, and text processing. It utilizes NLP models, such as SentenceTransformers and spaCy, to analyze threats and their relationships.

- **Imports and Model Initialization**: Loads necessary libraries and initializes models like `stsb-roberta-large` for semantic similarity, `spaCy` for NLP tasks, and WordNet for synset relations.

- **`semantic_similarity` Function**: Computes the cosine similarity between two sentences by encoding them with a pre-trained SentenceTransformer model.

- **`synset_relations` Function**: Extracts synset relations (e.g., hypernyms, hyponyms) for nouns in a sentence using WordNet.

- **`check_typeof_synset_relationship` Function**: Checks if two terms have a "type of" (hyponym/hypernym) relationship based on their WordNet synsets.

- **`check_partof_synset_relationship` Function**: Checks if two terms have a "part of" (meronym/holonym) relationship using WordNet synsets.

- **`compute_semantic_similarity_scores` Function**: Computes semantic similarity scores for each pair of sentences in the given input file. The results are saved in a CSV file.

- **`compute_semantic_similarity_scores_between_files` Function**: Computes semantic similarity scores for sentence pairs between two different files, saving the results to a new file.

- **`merge_csv_files` Function**: Merges two CSV files by adding identifiers to sentence pairs and writes the combined results to a new file.

- **`filter_dataframe_by_threshold` Function**: Filters a DataFrame to include only rows where the values in a specified column are greater than or equal to a given threshold.

- **`find_synset_relations` Function**: Analyses synset relationships between nouns in two sentence pairs (e.g., part-of or type-of relationships).

- **`analyse_sentence_voice` Function**: Analyses whether a sentence is in active or passive voice by inspecting its syntactic dependencies.

### Compute Tuple Similarity Scores

The script [compute_tuple_similarity_scores.py](https://github.com/tsumarios/TEAM/blob/main/compute_tuple_similarity_scores.py) computes the semantic similarity scores for tuples of cardinality *k* for both TEAM 2 and TEAM 3.
It uses a pre-trained model to encode and compare the semantic meaning of threats.

- **Imports and Model Initialisation**: Loads necessary libraries and initializes a pre-trained SentenceTransformer model (`stsb-roberta-large`) for computing semantic similarity.

- **`semantic_similarity` Function**: Computes cosine similarity between embeddings of two sentences for k=2, using SentenceTransformer’s utilities.

- **`compute_scores` Function**: Retrieves precomputed similarity scores for a given pair from a CSV file when k > 2.

- **`compute_combination_scores` Function**: For larger tuples (k > 2), this function calculates similarity scores between all sentence pairs within a tuple.

- **`compute_semantic_similarity_scores` Function**: Designed for k=2, this function iterates through all possible sentence pairs, computes similarity scores, and saves the results to the specified output CSV.

- **`save_scores_to_csv` Function**: Saves the scores for each tuple batch to CSV, appending data if the file already exists.

- **`compute_group_similarity_scores` Function**: The main execution logic. For k=2, it runs `compute_semantic_similarity_scores` directly. For k > 2, it:
  - Reads precomputed scores and iterates over combinations of sentences in tuples.
  - Computes and stores similarity statistics (max, mean, min).
  - Processes tuples in batches for efficiency and writes results to CSV.

- **`main` Function**: Parses command-line arguments (`--k`, `--in_path`, `--previous_k_scores_path`, `--out_path`) and initiates `compute_group_similarity_scores` with the given parameters.

## Data

The folder [data](https://github.com/tsumarios/TEAM/blob/main/data/) contains the inputs and outputs for TEAM 2 and TEAM 3, conveniently structured into subfolders.

For storage reasons, [TEAM 2](https://github.com/tsumarios/TEAM/tree/main/data/TEAM%202) only contains the semantic similarity scores computed for tuples with cardinality 3.

[TEAM 3](https://github.com/tsumarios/TEAM/tree/main/data/TEAM%203) contains the results of the first, second and third round, as well as the validation results. In particular, each of the iterations includes semantic relations (synset relations) for embraceable candidates. The validation results compare TEAM 3 with TEAM 1.

## Statistics and Images

The folder [img](https://github.com/tsumarios/TEAM/tree/main/img) provides some plots for both TEAM 2 and TEAM 3 statistics.

### PoC

The notebook [poc.ipynb](https://github.com/tsumarios/TEAM/blob/main/poc.ipynb) provides a Proof-of-Concept for the fundamentals behind the TEAM methods.

#### Contacts

- Email: <marioraciti@pm.me>
- LinkedIn: linkedin.com/in/marioraciti
- Twitter: twitter.com/tsumarios
