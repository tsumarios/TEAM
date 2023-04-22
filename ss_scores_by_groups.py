#!/usr/bin/env python3

'''
Step 4 of the Semi-Automated Embracing Approach Core
'''

import itertools
import pandas
from statistics import mean


preliminary_threats_path = './data/preliminary_threats.csv'
scores_path = './data/preliminary_threats_semantic_similarity_scores.csv'


def step4(preliminary_path: str, scores_path: str, k=3) -> tuple:
    '''
    Gathers sentences per groups of k elements, then retrieve the previously computed similarity scores.
    '''
    if k < 3:
        print('Please specify a cardinality greater or equal to 3.')
        return

    ss_df = pandas.read_csv(scores_path)
    sentence_list= pandas.read_csv(preliminary_path, index_col='P')['LB'].values.tolist()
    ordered_sentences = list(itertools.combinations(sentence_list, k))   # binomial coefficient yields the total number of groups, where n = len(sentence_list)

    scores_dict = {'max': [], 'mean': [], 'min': [], 'scores': []}
    for i in range(1, k+1):
        scores_dict[f'sentence{i}'] = []

    for tupl in ordered_sentences:
        pairs = list(itertools.combinations(tupl,2))
        scores = []
        for p in pairs:
            scores.append(ss_df.iloc[((ss_df['sentence1'].values==p[0]) & (ss_df['sentence2'].values==p[1])) |
                                     ((ss_df['sentence1'].values==p[1]) & (ss_df['sentence2'].values==p[0]))]['score'].values[0])
        for i in range(1, k+1):
            scores_dict[f'sentence{i}'].append(tupl[i-1])
        scores_dict['scores'].append(scores)
        scores_dict['max'].append(max(scores))
        scores_dict['mean'].append(mean(scores))
        scores_dict['min'].append(min(scores))
        break

    semantic_similarity_scores = pandas.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(f'{preliminary_path.split(".csv")[0]}_ss_scores_with_cardinality_{k}.csv', index=False, encoding='utf-8')


def main():
    for k in (4, 5, 6):
        step4(preliminary_threats_path, scores_path, k)


if __name__ == '__main__':
    main()
