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
from nltk.tokenize import word_tokenize
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
    common_hypernyms = set()
    result = False

    synsets1 = synsets(term1)
    synsets2 = synsets(term2)
    if not synsets1 or not synsets2:
        print('One or both terms do not have synsets in WordNet.')
        return False

    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 in list(synset2.closure(hypo, depth=3)):
                print(f'{term1} is a hyponym of {term2}')
                result = True
            elif synset1 in list(synset2.closure(hyper, depth=3)):
                print(f'{term1} is a hypernym of {term2}')
                result = True
            elif synset2 in list(synset1.closure(hypo, depth=3)):
                print(f'{term2} is a hyponym of {term1}')
                result = True
            elif synset2 in list(synset1.closure(hyper, depth=3)):
                print(f'{term2} is a hypernym of {term1}')
                result = True
            common_hypernyms.update(set(synset1.lowest_common_hypernyms(synset2)))

    if common_hypernyms:
        common_hypernym_names = list(set(hypernym.name().split('.')[0] for hypernym in common_hypernyms))
        print(f'{term1} and {term2} have common hypernyms: {", ".join(common_hypernym_names)}')
        result = True

    if result == False:
        print(f'{term1} and {term2} are not related ("type of")')
    return result


def check_partof_synset_relationship(term1: str, term2: str) -> bool:
    '''
    Check whether two terms have a "part of" synset relation.
    '''
    result = False

    synsets1 = synsets(term1)
    synsets2 = synsets(term2)
    if not synsets1 or not synsets2:
        print('One or both terms do not have synsets in WordNet.')
        return False

    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 in list(synset2.closure(part_holo, depth=3)):
                print(f'{term1} is a holonym of {term2}')
                result = True
            elif synset1 in list(synset2.closure(part_mero, depth=3)):
                print(f'{term1} is a meronym of {term2}')
                result = True
            elif synset2 in list(synset1.closure(part_holo, depth=3)):
                print(f'{term2} is a holonym of {term1}')
                result = True
            elif synset2 in list(synset1.closure(part_mero, depth=3)):
                print(f'{term2} is a meronym of {term1}')
                result = True

    if result == False:
        print(f'{term1} and {term2} are not related ("part of")')
    return result


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

    nouns1 = list(set([token.lemma_ for token in nlp(sentence1) if token.pos_ == 'NOUN']))
    nouns2 = list(set([token.lemma_ for token in nlp(sentence2) if token.pos_ == 'NOUN']))
    for term1 in nouns1:
        for term2 in nouns2:
            if term1 != term2:
                is_partof = check_partof_synset_relationship(term1, term2)
                is_typeof = check_typeof_synset_relationship(term1, term2)
            print()

    return is_partof, is_typeof
