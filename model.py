import wiki_text_parser as wtp 
from gensim import corpora
from gensim.parsing.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from smart_open import smart_open
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
import numpy as np  
from sklearn.feature_extraction import stop_words

def save_nlp_train_data(db_name, collection_name, target, fileout_dir):
    """use a mongodb collection and a subsampled target subset to create a .txt file
    with one line per document"""
    mc = MongoClient()
    db = mc[db_name]
    col = db[collection_name]
    target_pages = col.find({'target':target})
    df = pd.DataFrame(list(target_pages))['feature_union']
    subsampled_df = df.sample(frac=0.8, replace=False)
    with open(fileout_dir, 'w') as fout:
        for row in df:
            if row != 'nan':
                fout.write(row + '\n')

def create_trained_dictionary(filein, n_grams):
    """Use gensim to create a streamed dictionary"""
    # dictionary = [line for line in open(filein, 'r')]
    dictionary = corpora.Dictionary(_stem_and_ngramizer(line, n_grams) for line in open(filein, 'r'))
    dict_removed_extra = _remove_extra_words(dictionary)
    dict_removed_extra.save('nlp_training_data/math_dictionary.dict')
    return dict_removed_extra

def _remove_extra_words(dictionary):
    """remove words that appear only once"""
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq <= 2]
    dictionary.filter_tokens(once_ids)
    return dictionary

def _stem_and_ngramizer(line, n_grams):
    """stem all the words, generate ngrams, and return 
       a list of all stemmed words and ngram phrases"""
    p = PorterStemmer()
    s = SnowballStemmer('english')
    stopped = [word for word in line.split() if word not in stop_words.ENGLISH_STOP_WORDS]
    stems = [s.stem(word) for word in stopped] 
    grams = [[' '.join(stems[i:i+n]) for i in range(len(stems)-n+1)] for n in range(1, n_grams + 1)]
    return [item for sublist in grams for item in sublist]