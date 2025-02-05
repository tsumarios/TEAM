#!/usr/bin/env python3

"""
TEAM Core Script - Semantic Similarity Computation

This script calculates semantic similarity scores for threat tuples with configurable cardinality (k).
It supports both pairwise (k=2) and higher-order tuple combinations (k > 2), integrating with precomputed similarity scores for previous cardinalities.

Example Usage:
$ python compute_tuple_similarity_scores.py --k <k> --in_path ./data/input_threats.csv [--previous_k_scores_path ./data/input_threats_ss_scores.csv] --out_path ./data/output_similarity_scores.csv
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


def compute_semantic_similarity_matrix(sentences):
    """
    Compute the semantic similarity matrix for a list of sentences.
    """
    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    return similarity_matrix.cpu().numpy()


def compute_semantic_similarity_scores(in_path: str, out_path: str):
    """
    Compute semantic similarity scores for each pair of sentences (for k=2).
    """
    sentence_list = pd.read_csv(in_path, index_col="ID")["Threat"].values.tolist()
    similarity_matrix = compute_semantic_similarity_matrix(sentence_list)

    scores_dict = {"sentence1": [], "sentence2": [], "score": []}
    for i, sentence1 in enumerate(sentence_list):
        for j, sentence2 in enumerate(sentence_list):
            if i < j:
                scores_dict["sentence1"].append(sentence1)
                scores_dict["sentence2"].append(sentence2)
                scores_dict["score"].append(similarity_matrix[i, j])

    semantic_similarity_scores = pd.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(out_path, index=False, encoding="utf-8")

    min_score = semantic_similarity_scores["score"].round(2).min()
    max_score = semantic_similarity_scores["score"].round(2).max()
    return min_score, max_score


def compute_combination_scores(combination, ss_dict):
    """
    Computes similarity scores for a combination of sentences.
    """
    scores = []
    for i in range(len(combination)):
        for j in range(i + 1, len(combination)):
            key1 = (combination[i], combination[j])
            key2 = (combination[j], combination[i])

            score = ss_dict.get(key1, ss_dict.get(key2, None))
            if score is not None:
                scores.append(score)
    return scores


def compute_group_similarity_scores(
    in_path: str, previous_k_scores_path: str, out_path: str, k=3, start_time=None
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
    ss_df = pd.read_csv(previous_k_scores_path)
    ss_dict = {
        (row["sentence1"], row["sentence2"]): row["score"]
        for _, row in ss_df.iterrows()
    }
    sentence_list = pd.read_csv(in_path, usecols=["ID", "Threat"])["Threat"].tolist()
    ordered_sentences = itertools.combinations(sentence_list, k)
    sentences_len = sum(1 for _ in itertools.combinations(sentence_list, k))

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
                future = executor.submit(
                    compute_combination_scores, combination, ss_dict
                )
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
            batch_results = []

            for combination, future in futures:
                scores = future.result()
                batch_results.append(
                    {
                        **{f"sentence{i+1}": combination[i] for i in range(k)},
                        "scores": scores,
                        "max": max(scores),
                        "mean": np.mean(scores),
                        "min": min(scores),
                    }
                )

            batch_df = pd.DataFrame(batch_results)
            batch_df.to_csv(batch_file, index=False, encoding="utf-8")

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
        "--previous_k_scores_path",
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

    # Enforce previous_k_scores_path requirement when k > 2
    if args.k > 2 and args.previous_k_scores_path is None:
        parser.error("--previous_k_scores_path is required when --k > 2")

    start = datetime.now()
    compute_group_similarity_scores(
        args.in_path, args.previous_k_scores_path, args.out_path, args.k, start
    )
    end = datetime.now()
    print(f"Started at {start}\nFinished at {end}\nDelta {end-start}")


if __name__ == "__main__":
    main()
