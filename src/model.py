import src.wiki_db_parser as wdbp 
import src.page_disector as disector
import pandas as pd
import numpy as np 
import random
import sys
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from bson.objectid import ObjectId
from gensim import corpora, models, matutils
from gensim.sklearn_api import TfIdfTransformer
from gensim.parsing.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from smart_open import smart_open
from pymongo import MongoClient
from bson.objectid import ObjectId 
from sklearn.feature_extraction import stop_words
from timeit import default_timer 


random.seed(1)

def list_available_collections():
    return MongoClient()['wiki_cache'].list_collection_names()

def create_save_nlp_train_topic_model(db_name, collection_name, target, n_grams=3, subset=0.8):
    """with input of a category, creates a txt file of a SUBSET of the target,
        trains a dictionary, and trains a tfidf model. Saves these to files"""
    _save_txt_nlp_data(db_name, collection_name, target, training=True, subset=subset)
    _train_save_dictionary_corpus(f'nlp_training_data/{target}_subset.txt', n_grams, target, training=True)
    _train_save_tfidf(f'nlp_training_data/{target}_subset_corpus.mm', target, training=True)


def create_save_nlp_full_topic_model(db_name, collection_name, target, n_grams=3, subset=0.8):
    """with input of a category, creates a txt file of a FULL of the target,
        trains a dictionary, and trains a tfidf model. Saves these to files"""
    _save_txt_nlp_data(db_name, collection_name, target, training=False, subset=subset)
    _train_save_dictionary_corpus(f'nlp_training_data/{target}_full.txt', n_grams, target, training=False)
    _train_save_tfidf(f'nlp_training_data/{target}_full_corpus.mm', target, training=False)


def cross_validate_multinomial_nb(db_name, collection_name, target, n_grams=3):
    """train naive bayes model using a train/test split. Return predictions, score"""
    start = default_timer()
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    dictionary = corpora.Dictionary.load(f'nlp_training_data/{target}_subset.dict')
    tfidf = models.TfidfModel.load(f'nlp_training_data/{target}_subset.tfidf')
    print('CREATING stratified train/test split...')
    X_train_ids, X_test_ids, y_train, y_test = _get_train_test_ids(collection, 
                                                      target, test_percentage=0.8) 
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')    
    print('CREATING temporary txt file...')
    _make_temporary_txt(collection, X_train_ids)
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('CREATING training set bow with txt file')
    train_bow = [dictionary.doc2bow(word) for word in _list_grams('/tmp/docs_for_sparse_vectorization.txt', n_grams=n_grams)]
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('CREATING training set tfidf with txt file...')
    X_train_tfidf = tfidf[train_bow]
    pickle.dump(X_train_tfidf, open(f'nlp_training_data/{target}_X_train_tfidf.pkl', 'wb'))
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    y_test, preds, score = _fit_multinomial_nb(start, X_train_tfidf, y_train, y_test, db_name, collection, target, n_grams, X_test_ids, dictionary, tfidf)
    print(score)
    return y_test, preds, score, model
    

def _fit_multinomial_nb(start, X_train_tfidf, y_train, y_test, db_name, collection, target, n_grams, X_test_ids, dictionary, tfidf):
    """fit the actual model"""
    print('FITTING multinomial naive bayes model...')
    scipy_X_train = matutils.corpus2csc(X_train_tfidf).transpose()
    model = MultinomialNB().fit(scipy_X_train, y_train)#.reshape(-1, 1))
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print(f'Pickling model, saving to "nlp_training_data/{target}_multinomialNB_model.pkl"...')
    pickle.dump(model, open(f'nlp_training_data/{target}_multinomialNB_model.pkl', 'wb'))
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('CREATING test tfidf...')
    _make_temporary_txt(collection, X_test_ids)
    test_bow = [dictionary.doc2bow(word) for word in 
                     _list_grams('/tmp/docs_for_sparse_vectorization.txt',
                      n_grams=n_grams)]
    X_test_tfidf = tfidf[test_bow]
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('GENERATING predictions...')
    scipy_X_test = matutils.corpus2csc(X_test_tfidf).transpose()
    preds = model.predict_proba(scipy_X_test)
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('SCORING model...')
    score = log_loss(y_test, preds.T[1])
    print('DONE!')
    return y_test, preds, score, model


def cross_validate_logistic_regression(db_name, collection_name, target, n_grams=3):
    """train naive bayes model using a train/test split. Return predictions, score"""
    start = default_timer()
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    dictionary = corpora.Dictionary.load(f'nlp_training_data/{target}_subset.dict')
    tfidf = models.TfidfModel.load(f'nlp_training_data/{target}_subset.tfidf')
    print('CREATING stratified train/test split...')
    X_train_ids, X_test_ids, y_train, y_test = _get_train_test_ids(collection, 
                                                      target, test_percentage=0.8) 
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')    
    print('CREATING temporary txt file...')
    _make_temporary_txt(collection, X_train_ids)
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('CREATING training set bow with txt file')
    train_bow = [dictionary.doc2bow(word) for word in _list_grams('/tmp/docs_for_sparse_vectorization.txt', n_grams=n_grams)]
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('CREATING training set tfidf with txt file...')
    X_train_tfidf = tfidf[train_bow]
    pickle.dump(X_train_tfidf, open(f'nlp_training_data/{target}_X_train_tfidf.pkl', 'wb'))
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    y_test, preds, score = _fit_multinomial_nb(start, X_train_tfidf, y_train, y_test, db_name, collection, target, n_grams, X_test_ids, dictionary, tfidf)
    print(score)
    return y_test, preds, score, model
    

def _fit_logistic_regression(start, X_train_tfidf, y_train, y_test, db_name, collection, target, n_grams, X_test_ids, dictionary, tfidf):
    """fit the actual model"""
    print('FITTING multinomial naive bayes model...')
    scipy_X_train = matutils.corpus2csc(X_train_tfidf).transpose()
    model = LogisticRegression().fit(scipy_X_train, y_train)#.reshape(-1, 1))
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print(f'Pickling model, saving to "nlp_training_data/{target}_multinomialNB_model.pkl"...')
    pickle.dump(model, open(f'nlp_training_data/{target}_multinomialNB_model.pkl', 'wb'))
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('CREATING test tfidf...')
    _make_temporary_txt(collection, X_test_ids)
    test_bow = [dictionary.doc2bow(word) for word in 
                     _list_grams('/tmp/docs_for_sparse_vectorization.txt',
                      n_grams=n_grams)]
    X_test_tfidf = tfidf[test_bow]
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('GENERATING predictions...')
    scipy_X_test = matutils.corpus2csc(X_test_tfidf).transpose()
    preds = model.predict_proba(scipy_X_test)
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('SCORING model...')
    score = log_loss(y_test, preds.T[1])
    print('DONE!')
    return y_test, preds, score, model


def _save_txt_nlp_data(db_name, collection_name, target, training=True, subset=0.8):
    """use a mongodb collection and a subsampled target subset percentage
    to create a .txt file with one line per document"""
    print('Making txt file of subset of target class')
    mc = MongoClient()
    db = mc[db_name]
    col = db[collection_name]
    target_pages = col.find({'target':target})
    df = pd.DataFrame(list(target_pages))['feature_union']
    subsampled_df = df.sample(frac=subset, replace=False)
    if training:
        with open(f'nlp_training_data/{target}_subset.txt', 'w') as fout:
            for row in subsampled_df:
                if row != 'nan':
                    fout.write(row + '\n')
    else:
        with open(f'nlp_training_data/{target}_full.txt', 'w') as fout:
            for row in subsampled_df:
                if row != 'nan':
                    fout.write(row + '\n')
    print('DONE!')


def _train_save_dictionary_corpus(filein, n_grams, target, training=True):
    """Use gensim to create a streamed dictionary. 
    filein is the file used to train the dictionary and tfidf"""
    print('Building dictionary...')
    if training:
        dictionary = corpora.Dictionary(_list_grams(filein, n_grams))
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n = 100000)
        dictionary.save(f'nlp_training_data/{target}_subset.dict')
        corpus = [dictionary.doc2bow(word) for word in _list_grams(filein, n_grams)]
        corpora.MmCorpus.serialize(f'nlp_training_data/{target}_subset_corpus.mm', corpus)
    else:
        dictionary = corpora.Dictionary(_list_grams(filein, n_grams))
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n = 100000)
        dictionary.save(f'nlp_training_data/{target}_full.dict')
        corpus = [dictionary.doc2bow(word) for word in _list_grams(filein, n_grams)]
        corpora.MmCorpus.serialize(f'nlp_training_data/{target}_full_corpus.mm', corpus)
    print('DONE!')
    return dictionary, corpus


def _train_save_tfidf(filein, target, training=True):
    """input is a bow corpus saved as a tfidf file. The output is 
       a saved tfidf corpus"""
    print('Building TFIDF model')
    if training:
        try:
            corpus = corpora.MmCorpus(filein)
        except:
            raise NameError('HRMMPH. The file does not seem to exist. Create a file '+
                            'first by running the "train_save_dictionary_corpus" function.')
        tfidf = models.TfidfModel(corpus)
        tfidf.save(f'nlp_training_data/{target}_subset.tfidf')
        tfidf_corpus = tfidf[corpus]
    else:
        try:
            corpus = corpora.MmCorpus(filein)
        except:
            raise NameError('HRMMPH. The file does not seem to exist. Create a file '+
                            'first by running the "train_save_dictionary_corpus" function.')
        tfidf = models.TfidfModel(corpus)
        tfidf.save(f'nlp_training_data/{target}_full.tfidf')
        tfidf_corpus = tfidf[corpus]
    print('DONE!')
    return tfidf_corpus


def _make_temporary_txt(collection, ids):
    """make a temporary txt file of documents"""
    with open('/tmp/docs_for_sparse_vectorization.txt', 'w') as fout:
        for id in ids:
            res = collection.find_one(id) 
            text = res['feature_union']
            fout.write(text + '\n')

    
def _get_train_test_ids(collection, target, test_percentage=0.8):
    """get random train/test split, keeping the proportion of pos/neg
       classes the same"""
    pos_train_ids = []
    neg_train_ids = []
    pos_test_ids = []
    neg_test_ids = []
    pos_docs = collection.find({'target': target})
    neg_docs = collection.find({'target': {'$ne': target}})
    for doc in pos_docs:
        if random.random() < 0.8:
            pos_train_ids.append(doc['_id'])
        else:
            pos_test_ids.append(doc['_id'])
    for doc in neg_docs:
        if random.random() < 0.8:
            neg_train_ids.append(doc['_id'])
        else:
            neg_test_ids.append(doc['_id'])
    pos_train_y_list = list(np.ones(len(pos_train_ids)))
    neg_train_y_list = list(np.zeros(len(neg_train_ids)))
    pos_test_y_list = list(np.ones(len(pos_test_ids)))
    neg_test_y_list = list(np.zeros(len(neg_test_ids)))
    X_train_ids = pos_train_ids + neg_train_ids
    X_test_ids = pos_test_ids + neg_test_ids
    y_train = pos_train_y_list + neg_train_y_list
    y_test = pos_test_y_list + neg_test_y_list
    return X_train_ids, X_test_ids, np.array(y_train), np.array(y_test)
    

def _remove_extra_words(dictionary):
    """DEPRICATED! remove words that appear only once"""
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq <= 2]
    dictionary.filter_tokens(once_ids)
    return dictionary


def _list_grams(filein, n_grams):
    '''for each document, yield a list of strings'''
    with open(filein, 'r') as fin:
        for line in fin:
            yield _stem_and_ngramizer(line, n_grams)


def _stem_and_ngramizer(line, n_grams):
    """stem all the words, generate ngrams, and return 
       a list of all stemmed words and ngram phrases"""
    p = PorterStemmer()
    s = SnowballStemmer('english')
    stopped = [word for word in line.split() if word not in stop_words.ENGLISH_STOP_WORDS]
    stems = [s.stem(word) for word in stopped] 
    grams = [[' '.join(stems[i:i+n]) for i in range(len(stems)-n+1)] for n in range(1, n_grams + 1)]
    return [item for sublist in grams for item in sublist]


def main(arg1, arg2):
    pass


if __name__ == '__main__':
    if sys.argv != 3:
        raise NameError('Incorrect parameter count. Use: '+
                         'python model.py arg1 arg2')
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    main(arg1, arg2)