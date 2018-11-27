import wiki_text_parser as wtp 
from gensim import corpora
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
import numpy as np  

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
    df = pd.DataFrame(list(target_pages))
    return df
    subsampled_df = df.sample(frac=0.8, replace=False)
    with open(fileout_dir, 'w') as fout:
        for row in df:
            if row != 'nan':
                fout.write(row + '\n')

def create_trained_dictionary(filein):
    """Use gensim to create a streamed dictionary"""
    with open(filein, 'r') as fin:
        dictionary = corpora.Dictionary(line for line in fin.readline())
    return dictionary