#!/usr/bin/env python3

'''
Semi-Automated Embracing Approach Core
'''

# Imports and setup
import csv
import itertools
import nltk
import os
import pandas as pd
import spacy
import warnings
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util


warnings.filterwarnings('ignore')

# Wordnet
nltk.download('wordnet')
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


def compute_semantic_similarity_scores(in_path: str, out_path: str) -> tuple:
    '''
    Compute semantic similarity scores for each pair of sentences in the given input file.
    '''
    sentence_list= pd.read_csv(in_path, index_col='P')['LB'].values.tolist()
    ordered_sentences = list(itertools.combinations(sentence_list, 2))   # (n(n-1))/2 pairs, where n = len(sentence_list)


    scores_dict = {'sentence1': [], 'sentence2': [], 'score': []}
    for pair in ordered_sentences:
        # Get semantic similarity between sentence pairs
        similarity_score = semantic_similarity(pair[0], pair[1])
        scores_dict['sentence1'].append(pair[0])
        scores_dict['sentence2'].append(pair[1])
        scores_dict['score'].append(similarity_score)
    
    semantic_similarity_scores = pd.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(out_path, index=False, encoding='utf-8')

    min_score = semantic_similarity_scores['score'].round(2).min()
    max_score = semantic_similarity_scores['score'].round(2).max()
    return min_score, max_score


def compute_semantic_similarity_scores_between_files(in_path1: str, in_path2: str, out_path: str) -> tuple:
    '''
    Compute semantic similarity scores for each pair of sentences in the given input file.
    '''
    sentence_list1= pd.read_csv(in_path1, index_col='P')['LB'].values.tolist()
    sentence_list2= pd.read_csv(in_path2, index_col='P')['LB'].values.tolist()

    scores_dict = {'sentence1': [], 'score': []}
    for s1 in sentence_list1:
        max_ss_score = -1
        for s2 in sentence_list2:
            # Get semantic similarity between sentence pairs
            max_ss_score = max(max_ss_score, semantic_similarity(s1, s2))
        scores_dict['sentence1'].append(s1)
        scores_dict['score'].append(max_ss_score)

    semantic_similarity_scores = pd.DataFrame(scores_dict)
    semantic_similarity_scores.to_csv(out_path, index=False, encoding='utf-8')

    min_score = semantic_similarity_scores['score'].round(2).min()
    max_score = semantic_similarity_scores['score'].round(2).max()
    return min_score, max_score


def merge_csv_files(file1, file2, output_file):
    '''
    Merge preliminary threats with semantic similarity score to add identifiers.
    '''
    # Read the first CSV file and store the sentences with their identifiers
    sentences = {}
    with open(file1, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            identifier, sentence, _ = row
            sentences[sentence] = identifier

    # Create a new CSV file and write the header
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'id1', 'sentence1', 'id2', 'sentence2', 'score'])

        # Read the second CSV file and merge the data
        with open(file2, 'r') as f2:
            reader = csv.reader(f2)
            next(reader)  # Skip the header row
            for row in reader:
                index, sentence1, sentence2, score = row
                id1 = sentences.get(sentence1, '')
                id2 = sentences.get(sentence2, '')
                writer.writerow([index, id1, sentence1, id2, sentence2, score])
    
    # Remove old file2, now replaced by output_file
    os.remove(file2)
    os.rename(output_file, file2)


def filter_dataframe_by_threshold(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    '''
    Filter dataframe items where column values are greater or equal to given threshold.
    '''
    return df.loc[df[column] >= threshold]


def find_synset_relations(sentence1: str, sentence2: str) -> tuple:
    '''
    Look for synset relations between nouns in sentence pairs.
    '''
    is_partof, is_typeof = [False,], [False,]

    nouns1 = list(set([token.lemma_ for token in nlp(sentence1) if token.pos_ == 'NOUN']))
    nouns2 = list(set([token.lemma_ for token in nlp(sentence2) if token.pos_ == 'NOUN']))
    for term1 in nouns1:
        for term2 in nouns2:
            if term1 != term2:
                is_partof.append(check_partof_synset_relationship(term1, term2))
                is_typeof.append(check_typeof_synset_relationship(term1, term2))
            print()

    return any(is_partof), any(is_typeof)


def analyse_sentence_voice(sentence):
    '''
    Analyse a sentence voice (active/passive).
    '''
    sub, obj = None, None
    is_active, is_passive = False, False

    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
            sub = token.text
            is_active = True
        elif token.dep_ == 'nsubjpass' and token.head.pos_ == 'VERB':
            sub = token.text
            is_passive = True
        elif token.dep_ == 'dobj' and token.head.pos_ == 'VERB':
            obj = token.text
            is_active = True
        elif token.dep_ == 'pobj' and token.head.pos_ == 'VERB':
            obj = token.text
            is_passive = True

    return sub, obj, is_active, is_passive
