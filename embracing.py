#!/usr/bin/env python3

'''
Semi-Automated Embracing Approach Core
'''

# Imports and setup
import itertools
import pandas
import spacy
import warnings
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
from statistics import mean


warnings.filterwarnings('ignore')

# Semantic similarity model
model = SentenceTransformer('stsb-roberta-large')
# Spacy setup
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('merge_entities')

# Synset utility functions
hyper = lambda s: s.hypernyms()
hypo = lambda s: s.hyponyms()
part_mero = lambda s: s.part_meronyms()
part_holo = lambda s: s.part_holonyms()
synsets = lambda s: wn.synsets(s)


# Functions definition
def semantic_similarity(sentence1: str, sentence2: str) -> float:
    '''
    Calculate the semantic similarity between two sentences.
    '''
    # Encode threats to get their embeddings
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    # Compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_scores.item()


def synset_relations(sentence: str) -> dict:
    '''
    Get synset relations for each term within a sentence.
    '''
    result = {'sentence': sentence, 'terms': []}

    doc = nlp(sentence)
    # Get synset relations for nouns only
    nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
    for noun in nouns:
        # By default, consider the term as the first of the synonyms in the corpus.
        term = synsets(noun)[0] if synsets(noun) else None
        if term:
            # If the term is found, then retrieve the synset relations.
            relations = {
                'term': term,
                'synonyms': synsets(noun),
                'meronyms': list(term.closure(part_mero)),
                'holonyms': list(term.closure(part_holo)),
                'hypernyms [L1]': list(term.closure(hyper, depth=1)),
                'hypernyms [L2]': list(term.closure(hyper, depth=2)),
                'hypernyms [L3]': list(term.closure(hyper, depth=3)),
                'hyponyms [L1]': list(term.closure(hypo, depth=1)),
                'hyponyms [L2]': list(term.closure(hypo, depth=2)),
                'hyponyms [L3]': list(term.closure(hypo, depth=3))
            }
            result['terms'].append(relations)
        else:
            print(f'Term "{noun}" not found in corpus!')

    return result


def check_typeof_synset_relationship(term1: str, term2: str) -> bool:
    '''
    Check whether two terms have a "type of" synset relation.
    '''
    synsets1 = synsets(term1)
    synsets2 = synsets(term2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 in list(synset2.closure(hypo, depth=3)):
                print(f'{term1} is a hyponym of {term2}')
                return True
            elif synset1 in list(synset2.closure(hyper, depth=3)):
                print(f'{term1} is a hypernym of {term2}')
                return True
            elif synset2 in list(synset1.closure(hypo, depth=3)):
                print(f'{term2} is a hyponym of {term1}')
                return True
            elif synset2 in list(synset1.closure(hyper, depth=3)):
                print(f'{term2} is a hypernym of {term1}')
                return True
    print(f'{term1} and {term2} are not "type of" related')
    return False


def check_partof_synset_relationship(term1: str, term2: str) -> bool:
    '''
    Check whether two terms have a "part of" synset relation.
    '''
    synsets1 = synsets(term1)
    synsets2 = synsets(term2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 in list(synset2.closure(part_holo)):
                print(f'{term1} is a holonym of {term2}')
                return True
            elif synset1 in list(synset2.closure(part_mero)):
                print(f'{term1} is a meronym of {term2}')
                return True
            elif synset2 in list(synset1.closure(part_holo)):
                print(f'{term2} is a holonym of {term1}')
                return True
            elif synset2 in list(synset1.closure(part_mero)):
                print(f'{term2} is a meronym of {term1}')
                return True
    print(f'{term1} and {term2} are not "part of" related')
    return False


def step1(path: str) -> tuple:
    '''
    Compute semantic similarity scores for each pair of sentences in the given input file.
    '''
    sentence_list= pandas.read_csv(path, index_col='P')['LB'].values.tolist()
    ordered_sentences = list(itertools.combinations(sentence_list, 2))   # (n(n-1))/2 pairs, where n = len(sentence_list)


    scores_dict = {'sentence1': [], 'sentence2': [], 'score': []}
    for pair in ordered_sentences:
        # Get semantic similarity between sentence pairs
        similarity_score = semantic_similarity(pair[0], pair[1])
        scores_dict['sentence1'].append(pair[0])
        scores_dict['sentence2'].append(pair[1])
        scores_dict['score'].append(similarity_score)
    
    semantic_similarity_scores = pandas.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(f'{path.split(".csv")[0]}_semantic_similarity_scores.csv', index=False, encoding='utf-8')

    min_score = semantic_similarity_scores['score'].round(2).min()
    max_score = semantic_similarity_scores['score'].round(2).max()
    return min_score, max_score


def step2(path: str, threshold: float) -> pandas.DataFrame:
    '''
    Filter semantic similarity scores dataframe greater or equal to given threshold.
    '''
    semantic_similarity_scores = pandas.read_csv(path)
    return semantic_similarity_scores.loc[semantic_similarity_scores['score'] >= threshold]


def step3(sentence1: str, sentence2: str) -> tuple:
    '''
    Look for synset relations between nouns in sentence pairs.
    '''
    is_partof, is_typeof = False, False

    synset1 = synset_relations(sentence1)
    synset2 = synset_relations(sentence2)
    for term1 in synset1.get('terms', []):
        for term2 in synset2.get('terms', []):
            if term1 != term2:
                for s1 in term1.get('synonyms'):
                    for s2 in term2.get('synonyms'):
                        if s1 in list(s2.closure(hypo, depth=3)):
                            print(f'{s1.name()} is a hyponym of {s2.name()}')
                            is_typeof = True
                        elif s1 in list(s2.closure(hyper, depth=3)):
                            print(f'{s1.name()} is a hypernym of {s2.name()}')
                            is_typeof = True
                        elif s2 in list(s1.closure(hypo, depth=3)):
                            print(f'{s2.name()} is a hyponym of {s1.name()}')
                            is_typeof = True
                        elif s2 in list(s1.closure(hyper, depth=3)):
                            print(f'{s2.name()} is a hypernym of {s1.name()}')
                            is_typeof = True
                        if s1 in list(s2.closure(part_holo)):
                            print(f'{s1.name()} is a holonym of {s2.name()}')
                            is_partof = True
                        elif s1 in list(s2.closure(part_mero)):
                            print(f'{s1.name()} is a meronym of {s2.name()}')
                            is_partof = True
                        elif s2 in list(s1.closure(part_holo)):
                            print(f'{s2.name()} is a holonym of {s1.name()}')
                            is_partof = True
                        elif s2 in list(s1.closure(part_mero)):
                            print(f'{s2.name()} is a meronym of {s1.name()}')
                            is_partof = True

    return is_partof, is_typeof


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

    semantic_similarity_scores = pandas.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(f'{preliminary_path.split(".csv")[0]}_ss_scores_with_cardinality_{k}.csv', index=False, encoding='utf-8')
