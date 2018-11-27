import wiki_text_parser as wtp 
from gensim import corpora
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd

def get_nlp_train_data(db_name, collection_name, target):
    mc = MongoClient()
    db = mc[db_name]
    col = db[collection_name]
    target_pages = col.find({'target':target})
    df = pd.DataFrame(list(target_pages))['full_raw_xml']
    subsampled_df = df.sample(frac=0.8, replace=True)
    return subsampled_df.tolist()



