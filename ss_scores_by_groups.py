#!/usr/bin/env python3

'''
Compute Similarity Scores for tuples with cardinality k - Semi-Automated Embracing Approach Core
'''

import os
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime


preliminary_threats_path = './data/preliminary_threats.csv'
scores_path = './data/preliminary_threats_semantic_similarity_scores.csv'


def compute_scores(pair, ss_df):
    '''
    Computes the similarity score for a pair of sentences.
    '''
    mask = ((ss_df['sentence1'] == pair[0]) & (ss_df['sentence2'] == pair[1])) | \
           ((ss_df['sentence1'] == pair[1]) & (ss_df['sentence2'] == pair[0]))
    return ss_df.loc[mask, 'score'].values[0]


def compute_combination_scores(combination, ss_df):
    '''
    Computes the similarity scores for a batch of sentence combinations.
    '''
    pairs = np.array(list(itertools.combinations(combination, 2)))
    scores = np.array([compute_scores(pair, ss_df) for pair in pairs])
    return scores


def save_scores_to_csv(scores_dict, output_file):
    '''
    Saves scores to a CSV file.
    '''
    semantic_similarity_scores = pd.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False, encoding='utf-8')


def step4(preliminary_path: str, scores_path: str, k=3, start_time=None) -> None:
    '''
    Gathers sentences per groups of k elements, then retrieve the previously computed similarity scores.
    '''
    if k < 3:
        print('Please specify a cardinality greater or equal to 3.')
        return

    ss_df = pd.read_csv(scores_path)
    sentence_list = pd.read_csv(preliminary_path, index_col='P')['LB'].values.tolist()
    ordered_sentences = itertools.combinations(sentence_list, k)

    scores_dict = {'max': [], 'mean': [], 'min': [], 'scores': []}
    sentence_keys = [f'sentence{i}' for i in range(1, k + 1)]
    scores_dict.update({key: [] for key in sentence_keys})

    j = 0   # Just for statistics
    batch_size = 1000  # Adjust the batch size for optimal performance
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
                scores_dict['scores'].append(scores)
                scores_dict['max'].append(max(scores))
                scores_dict['mean'].append(np.mean(scores))
                scores_dict['min'].append(min(scores))

            # Save scores to a batch file
            output_file = f'{preliminary_path.split(".csv")[0]}_ss_scores_with_cardinality_{k}_batch{j}.csv'
            save_scores_to_csv(scores_dict, output_file)

            # Clear scores_dict to free up memory
            scores_dict = {'max': [], 'mean': [], 'min': [], 'scores': []}
            scores_dict.update({key: [] for key in sentence_keys})

            combinations_batch = list(itertools.islice(ordered_sentences, batch_size))

            # Print statistics
            now = datetime.now()
            print(f'{now} - Iteration {j} of 57,941')
            print(f'Elapsed time {now-start_time}')
            j += 1

    # Merge batch files
    output_file = f'{preliminary_path.split(".csv")[0]}_ss_scores_with_cardinality_{k}.csv'
    with open(output_file, 'w') as output_csv:
        for batch_index in range(j):
            batch_file = f'{preliminary_path.split(".csv")[0]}_ss_scores_with_cardinality_{k}_batch{batch_index}.csv'
            with open(batch_file, 'r') as batch_csv:
                if batch_index > 0:
                    next(batch_csv)  # Skip the header of subsequent batch files
                output_csv.write(batch_csv.read())

            os.remove(batch_file)  # Remove the merged batch file


def main():
    k = 6   # NOTE: Change me!
    start = datetime.now()
    step4(preliminary_threats_path, scores_path, k, start)
    end = datetime.now()
    print(f'Started at {start}\nFinished at {end}\nDelta {end-start}')


if __name__ == '__main__':
    main()
