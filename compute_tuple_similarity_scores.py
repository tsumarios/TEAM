#!/usr/bin/env python3

"""
TEAM Core Script - Semantic Similarity Computation

This script calculates semantic similarity scores for threat tuples with configurable cardinality (k).
It supports both pairwise (k=2) and higher-order tuple combinations (k > 2), integrating with precomputed similarity scores for previous cardinalities.

Example Usage:
$ python compute_tuple_similarity_scores.py --k <k> --in_path ./data/input_threats.csv [--scores_path ./data/input_threats_ss_scores.csv] --out_path ./data/output_similarity_scores.csv
"""

import os
import itertools
import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from sentence_transformers import SentenceTransformer, util


# Semantic similarity model
model = SentenceTransformer("stsb-roberta-large")


def semantic_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the semantic similarity between two sentences.
    """
    # Encode threats to get their embeddings
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    # Compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_scores.item()


def compute_scores(pair, ss_df):
    """
    Computes the similarity score for a pair of sentences.
    """
    mask = ((ss_df["sentence1"] == pair[0]) & (ss_df["sentence2"] == pair[1])) | (
        (ss_df["sentence1"] == pair[1]) & (ss_df["sentence2"] == pair[0])
    )
    return ss_df.loc[mask, "score"].values[0]


def compute_combination_scores(combination, ss_df):
    """
    Computes the similarity scores for a batch of sentence combinations.
    """
    pairs = np.array(list(itertools.combinations(combination, 2)))
    scores = np.array([compute_scores(pair, ss_df) for pair in pairs])
    return scores


def compute_semantic_similarity_scores(in_path: str, out_path: str):
    """
    Compute semantic similarity scores for each pair of sentences (for k=2).
    """
    sentence_list = pd.read_csv(in_path, index_col="ID")["Threat"].values.tolist()
    ordered_sentences = list(itertools.combinations(sentence_list, 2))

    scores_dict = {"sentence1": [], "sentence2": [], "score": []}
    for pair in ordered_sentences:
        similarity_score = semantic_similarity(pair[0], pair[1])
        scores_dict["sentence1"].append(pair[0])
        scores_dict["sentence2"].append(pair[1])
        scores_dict["score"].append(similarity_score)

    semantic_similarity_scores = pd.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(out_path, index=False, encoding="utf-8")

    min_score = semantic_similarity_scores["score"].round(2).min()
    max_score = semantic_similarity_scores["score"].round(2).max()
    return min_score, max_score


def save_scores_to_csv(scores_dict, output_file):
    """
    Saves scores to a CSV file.
    """
    semantic_similarity_scores = pd.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(
        output_file,
        mode="a",
        header=not os.path.exists(output_file),
        index=False,
        encoding="utf-8",
    )


def compute_group_similarity_scores(
    in_path: str, scores_path: str, out_path: str, k=3, start_time=None
):
    """
    Computes similarity scores for tuples of cardinality k. Uses a direct method for k=2.
    """
    if k == 2:
        print("Computing pairwise similarity scores (k=2)...")
        min_score, max_score = compute_semantic_similarity_scores(in_path, out_path)
        print(f"Completed: min score={min_score}, max score={max_score}")
        return

    print(f"Computing similarity scores for k={k}...")
    ss_df = pd.read_csv(scores_path)
    sentence_list = pd.read_csv(in_path, index_col="ID")["Threat"].values.tolist()
    ordered_sentences = itertools.combinations(sentence_list, k)
    sentences_len = len(list(itertools.combinations(sentence_list, k)))

    scores_dict = {"max": [], "mean": [], "min": [], "scores": []}
    sentence_keys = [f"sentence{i}" for i in range(1, k + 1)]
    scores_dict.update({key: [] for key in sentence_keys})

    j = 0  # Statistics counter
    batch_size = 1000
    combinations_batch = list(itertools.islice(ordered_sentences, batch_size))

    with ProcessPoolExecutor(max_workers=8) as executor:
        while combinations_batch:
            futures = []
            for combination in combinations_batch:
                future = executor.submit(compute_combination_scores, combination, ss_df)
                futures.append((combination, future))

            for combination, future in futures:
                scores = future.result()

                # Store results in scores_dict
                for i, sentence in enumerate(combination):
                    scores_dict[sentence_keys[i]].append(sentence)
                scores_dict["scores"].append(scores)
                scores_dict["max"].append(max(scores))
                scores_dict["mean"].append(np.mean(scores))
                scores_dict["min"].append(min(scores))

            # Save scores to a batch file
            batch_file = (
                f'{out_path.split(".csv")[0]}_with_cardinality_{k}_batch{j}.csv'
            )
            save_scores_to_csv(scores_dict, batch_file)

            # Clear scores_dict to free up memory
            scores_dict = {"max": [], "mean": [], "min": [], "scores": []}
            scores_dict.update({key: [] for key in sentence_keys})

            combinations_batch = list(itertools.islice(ordered_sentences, batch_size))

            # Print statistics
            now = datetime.now()
            print(f"{now} - Iteration {j} of {round(sentences_len/batch_size)}")
            print(f"Elapsed time {now-start_time}")
            j += 1

    # Merge batch files
    with open(out_path, "w") as output_csv:
        for batch_index in range(j):
            batch_file = f'{out_path.split(".csv")[0]}_with_cardinality_{k}_batch{batch_index}.csv'
            with open(batch_file, "r") as batch_csv:
                if batch_index > 0:
                    next(batch_csv)  # Skip header for subsequent batches
                output_csv.write(batch_csv.read())
            os.remove(batch_file)  # Remove batch file


def main():
    parser = argparse.ArgumentParser(
        description="Compute semantic similarity scores for threat tuples."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Cardinality of tuples (e.g., 2 for pairs, >2 for larger tuples).",
    )
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to the input threats file (CSV).",
    )
    parser.add_argument(
        "--scores_path",
        type=str,
        required=False,  # required only if k > 2
        help="Path to the precomputed semantic similarity scores file (CSV) for k-1. Needed only if k > 2.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to save the output scores (CSV).",
    )

    args = parser.parse_args()

    start = datetime.now()
    compute_group_similarity_scores(
        args.in_path, args.scores_path, args.out_path, args.k, start
    )
    end = datetime.now()
    print(f"Started at {start}\nFinished at {end}\nDelta {end-start}")


if __name__ == "__main__":
    main()
