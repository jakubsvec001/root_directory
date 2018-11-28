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

# class MyCorpus(object):
#     """a generator class for reading pregenerated nlp training data, line by line"""

#     def __iter__(self):
#         with open('seed/training_mathematics_text', 'r') as fin:
#             for line in fin:
#             # assume there's one document per line, tokens separated by whitespace
#                 yield dictionary.doc2bow(line.lower().split())

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

def create_trained_dictionary(filein):
    """Use gensim to create a streamed dictionary"""
    # dictionary = [line for line in open(filein, 'r')]
    p = PorterStemmer()
    s = SnowballStemmer('english')
    dictionary = corpora.Dictionary([s.stem(word) for word in line.split()] for line in open(filein, 'r'))
    dict_removed_extra = remove_extra_words(dictionary)
    dict_removed_extra.save('nlp_training_data/math_dictionary.dict')
    return dict_removed_extra

def remove_extra_words(dictionary):
    """remove stopwords and words that appear only once"""
    stop_ids = [dictionary.token2id[stopword] for stopword in stop_words.ENGLISH_STOP_WORDS
                if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)
    return dictionary