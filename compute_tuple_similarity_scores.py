#!/usr/bin/env python3

"""
TEAM Core Script - Semantic Similarity Computation

This script calculates semantic similarity scores for threat tuples with configurable cardinality (k).
It supports both pairwise (k=2) and higher-order tuple combinations (k > 2).

Example Usage:
$ python compute_tuple_similarity_scores.py --k <k> --in_path ./data/input_threats.csv --out_path ./data/output_similarity_scores.csv
"""

import os
import itertools
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Semantic similarity model
model = SentenceTransformer("stsb-roberta-large", device="cpu")


def compute_semantic_similarity_matrix(sentences):
    """
    Compute the semantic similarity matrix for a list of sentences.
    """
    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
    return similarity_matrix


def compute_semantic_similarity_scores(in_path: str, out_path: str):
    """
    Compute semantic similarity scores for each pair of sentences (for k=2).
    """
    sentence_list = pd.read_csv(in_path, index_col="ID")["Threat"].values.tolist()
    similarity_matrix = compute_semantic_similarity_matrix(sentence_list)

    # Use NumPy indexing to extract upper-triangle scores
    i_upper, j_upper = np.triu_indices(len(sentence_list), k=1)
    scores = similarity_matrix[i_upper, j_upper]

    semantic_similarity_scores = pd.DataFrame(
        {
            "sentence1": [sentence_list[i] for i in i_upper],
            "sentence2": [sentence_list[j] for j in j_upper],
            "score": scores,
        }
    )
    semantic_similarity_scores.to_csv(out_path, index=False, encoding="utf-8")
    return scores.min(), scores.max()


def compute_tuple_scores(similarity_matrix, tuple_indices):
    """
    Compute tuple similarity scores using matrix indexing.
    """
    pairwise_scores = similarity_matrix[np.ix_(tuple_indices, tuple_indices)]
    pairwise_scores = pairwise_scores[np.triu_indices(len(tuple_indices), k=1)]
    return {
        "scores": pairwise_scores.tolist(),
        "max": np.max(pairwise_scores),
        "mean": np.mean(pairwise_scores),
        "min": np.min(pairwise_scores),
    }


def compute_group_similarity_scores(in_path: str, out_path: str, k=3, start_time=None):
    """
    Computes similarity scores for tuples of cardinality k. Uses a direct method for k=2.
    """
    if k == 2:
        print("Computing pairwise similarity scores (k=2)...")
        min_score, max_score = compute_semantic_similarity_scores(in_path, out_path)
        print(f"Completed: min score={min_score}, max score={max_score}")
        return

    print(f"Computing similarity scores for k={k}...")
    df = pd.read_csv(in_path, usecols=["ID", "Threat"])
    sentence_list = df["Threat"].tolist()
    similarity_matrix = compute_semantic_similarity_matrix(sentence_list)

    tuple_combinations = list(itertools.combinations(range(len(sentence_list)), k))
    total_combinations = len(tuple_combinations)
    scores_list = []

    for i, tuple_indices in enumerate(tuple_combinations):
        scores = compute_tuple_scores(similarity_matrix, tuple_indices)
        scores_list.append(
            {
                **{
                    f"sentence{i+1}": sentence_list[idx]
                    for i, idx in enumerate(tuple_indices)
                },
                "scores": scores["scores"],
                "max": scores["max"],
                "mean": scores["mean"],
                "min": scores["min"],
            }
        )

        # Calculate progress percentage and ETA
        progress = (i + 1) / total_combinations * 100
        elapsed_time = datetime.now() - start_time
        elapsed_seconds = elapsed_time.total_seconds()  # Convert to seconds
        eta = (elapsed_seconds / (i + 1)) * (total_combinations - (i + 1))

        # Display progress and ETA
        if (i + 1) % max(
            1, total_combinations // 10
        ) == 0 or i == total_combinations - 1:
            print(
                f"Progress: {progress:.2f}% | Elapsed Time: {elapsed_seconds:.2f}s | ETA: {eta:.2f}s"
            )

    pd.DataFrame(scores_list).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(scores_list)} tuple scores to {out_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute semantic similarity scores for threat tuples."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Tuple cardinality (2 for pairs, >2 for larger tuples).",
    )
    parser.add_argument(
        "--in_path", type=str, required=True, help="Path to input threats file (CSV)."
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to save output scores (CSV)."
    )

    args = parser.parse_args()

    start = datetime.now()
    compute_group_similarity_scores(args.in_path, args.out_path, args.k, start)
    end = datetime.now()
    print(f"Started at {start}\nFinished at {end}\nDelta {end-start}")


if __name__ == "__main__":
    main()
