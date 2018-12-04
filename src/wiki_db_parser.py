from bs4 import BeautifulSoup as bs
import src.page_disector as disector
from bson.objectid import ObjectId
import mwparserfromhell
import re
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import glob
from pymongo import MongoClient
from gensim.corpora import wikicorpus
import sys


def disect_update_database(db_name, collection_name):
    """parse and update a mongodb database with cleaned xml"""
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    document_generator = mongodb_page_stream(collection)
    count = 0
    for document in document_generator:
        features = disector.disect_page(document['title'], document['full_raw_xml'])
        _update_document(collection, features)
        count +=1
        sys.stdout.write('\r'+ f'YAAAAAAAS! Updated {count} documents!')


def mongodb_page_stream(collection):
    """yield a new page from a mongodb collection"""
    document_generator = collection.find()
    return document_generator


def convert_wiki_cache_to_edgelist(collection_name_out, db_name='wiki_cache', new_collection='small_edgelist'):
    """Use wiki_cache database and convert it to a small edgelist collection"""
    mc = MongoClient()
    db = mc[db_name]
    collection_out = db[collection_name_out]
    collection_in = db[new_collection]
    wiki_cache_pages = collection_out.find()
    for i, page in enumerate(wiki_cache_pages):
        sys.stdout.write('\r' + f'Wrote {i} pages to {new_collection}.')
        page = page['parent_categories']
        title = page[0]
        categories = page[1]
        cached_child = collection_in.find_one({'parent_categories': [title, categories]})
        if cached_child is None:
            collection_in.insert_one({'parent_categories': [title, categories]})
    

def _update_document(collection, features):
    """Used by disect_update_database(). Updates a document in mongodb"""
    collection.update_one({'title': features['title']}, {'$set': 
                                                           {'feature_union': features['feature_union'],
                                                            'parent_categories': features['parent_categories']}})


def num_target_col_db(db_name, collection_name, target_name):
    """add a column to each document with a number and the target value
       for indexing train/test split. Uses _check_target_presence_in_db()"""
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    _check_target_presence_in_db(collection, target_name)
    docs = collection.find()
    for idx, doc in enumerate(docs):
        _id = doc['_id']
        if doc['target'] == target_name:
            target = 1
        else:
            target = 0
        collection.update({'_id': ObjectId(_id)}, {'$set' : 
                                                            {'idx/target': (idx, target)}})


def _check_target_presence_in_db(collection, target_name):
    """check for presence of target in db 
    before adding num_target column"""
    docs = collection.find()
    hit = False
    for doc in docs:
        if doc['target'] == target_name:
            hit = True
            break
    if hit == False:
        raise ValueError('OH NOOO! No targets saved. '+
                         'Check your target_name parameter')


if __name__ == '__main__':
    pass