#!/usr/bin/env python3

'''
Compute Similarity Scores for tuples with cardinality k - Semi-Automated Embracing Approach Core
'''

import itertools
import pandas as pd
from datetime import datetime
from statistics import mean
from concurrent.futures import ThreadPoolExecutor


preliminary_threats_path = './data/preliminary_threats.csv'
scores_path = './data/preliminary_threats_semantic_similarity_scores.csv'


def compute_scores(pair, ss_df):
    """
    Computes the similarity score for a pair of sentences.
    """
    return ss_df.loc[
        ((ss_df['sentence1'] == pair[0]) & (ss_df['sentence2'] == pair[1])) |
        ((ss_df['sentence1'] == pair[1]) & (ss_df['sentence2'] == pair[0]))
    ]['score'].values[0]


def compute_combination_scores(combinations, ss_df):
    """
    Computes the similarity scores for a batch of sentence combinations.
    """
    scores = []
    for combination in combinations:
        pairs = list(itertools.combinations(combination, 2))
        combination_scores = [compute_scores(pair, ss_df) for pair in pairs]
        scores.append(combination_scores)
    return scores


def step4(preliminary_path: str, scores_path: str, k=3) -> None:
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

    print('Starting:')
    print(scores_dict)

    with ThreadPoolExecutor() as executor:
        batch_size = 50000  # Adjust the batch size for optimal performance
        combinations_batch = list(itertools.islice(ordered_sentences, batch_size))

        j = 0
        while combinations_batch:
            futures = []
            for combination in combinations_batch:
                future = executor.submit(compute_combination_scores, [combination], ss_df)
                futures.append((combination, future))

            for combination, future in futures:
                scores = future.result()[0]

                for i, sentence in enumerate(combination):
                    scores_dict[sentence_keys[i]].append(sentence)

                scores_dict['scores'].append(scores)
                scores_dict['max'].append(max(scores))
                scores_dict['mean'].append(mean(scores))
                scores_dict['min'].append(min(scores))

            combinations_batch = list(itertools.islice(ordered_sentences, batch_size))
            print(f'Iteration {j} of 1158')
            j = j + 1

    semantic_similarity_scores = pd.DataFrame(scores_dict)
    output_file = f'{preliminary_path.split(".csv")[0]}_ss_scores_with_cardinality_{k}.csv'
    semantic_similarity_scores.to_csv(output_file, index=False, encoding='utf-8')


def main():
    k = 5   # NOTE: Change me!
    start = datetime.now()
    step4(preliminary_threats_path, scores_path, k)
    end = datetime.now()
    print(f'Started at {start}\nFinished at {end}')


if __name__ == '__main__':
    main()
