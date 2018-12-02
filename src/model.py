import src.wiki_db_parser as wdbp 
import src.page_disector as disector
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import random
import sys
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_curve, auc
from bson.objectid import ObjectId
from scipy import interp
from gensim import corpora, models, matutils
from gensim.sklearn_api import TfIdfTransformer
from gensim.parsing.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from smart_open import smart_open
from pymongo import MongoClient
from bson.objectid import ObjectId 
from sklearn.feature_extraction import stop_words
from timeit import default_timer 


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
    _, X_train_ids, X_test_ids, y_train, y_test = _get_train_test_ids(collection, 
                                                      target, train_percentage=0.8) 
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
    y_test, preds, score, model = _fit_multinomial_nb(start, X_train_tfidf, y_train, y_test, db_name, collection, target, n_grams, X_test_ids, dictionary, tfidf)
    print(score)
    return y_test, preds, score, model
    

def _fit_multinomial_nb(start, X_train_tfidf, y_train, y_test, db_name, collection, target, n_grams, X_test_ids, dictionary, tfidf):
    """fit the actual model"""
    print('FITTING multinomial naive bayes model...')
    scipy_X_train = matutils.corpus2csc(X_train_tfidf).transpose()
    model = MultinomialNB().fit(scipy_X_train, y_train)
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


def k_fold_logistic_regression(db_name, collection_name, target, C, feature_count, n_grams=3, k_folds=5, seed=None):
    """train naive bayes model using a train/test split. Return predictions, score"""
    start = default_timer()
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    dictionary = corpora.Dictionary.load(f'nlp_training_data/{target}_subset.dict')
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n = feature_count)
    tfidf = models.TfidfModel.load(f'nlp_training_data/{target}_subset.tfidf')
    k_fold_ids = _get_k_fold_ids(collection, target, seed, k_folds)  
    model_list = []
    y_test_list = []
    pred_list = []
    score_list = []
    for i, X_train_ids, X_test_ids, y_train, y_test in k_fold_ids:
        print(f'RUNNING K_FOLD #{i+1} OF {k_folds}...')
        print('    CREATING temporary txt file...')
        _make_temporary_txt(collection, X_train_ids)
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    CREATING training set bow with txt file')
        train_bow = [dictionary.doc2bow(word) for word in _list_grams('/tmp/docs_for_sparse_vectorization.txt', n_grams=n_grams)]
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    CREATING training set tfidf with txt file...')
        X_train_tfidf = tfidf[train_bow]
        pickle.dump(X_train_tfidf, open(f'nlp_training_data/{target}_X_train_tfidf.pkl', 'wb'))
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    FITTING logistic regression model...')
        scipy_X_train = matutils.corpus2csc(X_train_tfidf).transpose()
        model = LogisticRegression(penalty='l2', solver='saga', C=C).fit(scipy_X_train, y_train)
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    CREATING test tfidf...')
        _make_temporary_txt(collection, X_test_ids)
        test_bow = [dictionary.doc2bow(word) for word in 
                        _list_grams('/tmp/docs_for_sparse_vectorization.txt',
                        n_grams=n_grams)]
        X_test_tfidf = tfidf[test_bow]
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    GENERATING predictions...')
        scipy_X_test = matutils.corpus2csc(X_test_tfidf).transpose()
        predictions = model.predict_proba(scipy_X_test)
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    SCORING model...')
        score = log_loss(y_test, predictions.T[1])
        print('DONE!')
        model_list.append(model)
        y_test_list.append(y_test)
        pred_list.append(predictions)
        score_list.append(score)
        print(f'score: {score}')
    _plot_roc_curves(y_test_list, pred_list, C, feature_count)
    return score_list, y_test_list, pred_list, model_list

def grid_search_logistic_regression(db_name, collection_name, target, C, feature_count=100000, n_grams=3, k_folds=5, seed=None):
    """gridsearch without cross validation"""
    start = default_timer()
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    dictionary = corpora.Dictionary.load(f'nlp_training_data/{target}_subset.dict')
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n = feature_count)
    tfidf = models.TfidfModel.load(f'nlp_training_data/{target}_subset.tfidf')
    i, X_train_ids, X_test_ids, y_train, y_test = _get_train_test_ids(collection, 
                                                                        target, 
                                                                        train_percentage=0.8, 
                                                                        seed=None)  
    model_list = []
    y_test_list = []
    pred_list = []
    score_list = []
    for c in C:
        print(f'RUNNING GRIDSEARCH FOR {c}...')
        print('    CREATING temporary txt file...')
        _make_temporary_txt(collection, X_train_ids)
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    CREATING training set bow with txt file')
        train_bow = [dictionary.doc2bow(word) for word in _list_grams('/tmp/docs_for_sparse_vectorization.txt', n_grams=n_grams)]
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    CREATING training set tfidf with txt file...')
        X_train_tfidf = tfidf[train_bow]
        pickle.dump(X_train_tfidf, open(f'nlp_training_data/{target}_X_train_tfidf.pkl', 'wb'))
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    FITTING logistic regression model...')
        scipy_X_train = matutils.corpus2csc(X_train_tfidf).transpose()
        model = LogisticRegression(penalty='l2', solver='saga', C=c).fit(scipy_X_train, y_train)
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    CREATING test tfidf...')
        _make_temporary_txt(collection, X_test_ids)
        test_bow = [dictionary.doc2bow(word) for word in 
                        _list_grams('/tmp/docs_for_sparse_vectorization.txt',
                        n_grams=n_grams)]
        X_test_tfidf = tfidf[test_bow]
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    GENERATING predictions...')
        scipy_X_test = matutils.corpus2csc(X_test_tfidf).transpose()
        predictions = model.predict_proba(scipy_X_test)
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    SCORING model...')
        score = log_loss(y_test, predictions.T[1])
        print('DONE!')
        model_list.append(model)
        y_test_list.append(y_test)
        pred_list.append(predictions)
        score_list.append(score)
        print(f'score: {score}')
        _plot_roc_curves(y_test_list, pred_list, c, feature_count)
    return score_list, y_test_list, pred_list, model_list

def _plot_roc_curves(y_test_list, pred_list, C, feature_count):
    """plot roc curve for each cross_validated model"""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(1,1, figsize=((20,20)))
    for i in range(len(y_test_list)):
        fpr, tpr, thresholds = roc_curve(y_test_list[i], pred_list[i][:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        print()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Logistic Regression ROC, C={round(C, 3)}, Feature Count={feature_count}')
    ax.legend(loc="lower right")
    fig.savefig(f'images/roc_cv_logistic_regression_{round(C, 3)}_{feature_count}.png')
    fig.show()


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
        dictionary.save(f'nlp_training_data/{target}_subset.dict')
        corpus = [dictionary.doc2bow(word) for word in _list_grams(filein, n_grams)]
        corpora.MmCorpus.serialize(f'nlp_training_data/{target}_subset_corpus.mm', corpus)
    else:
        dictionary = corpora.Dictionary(_list_grams(filein, n_grams))
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

    
def _get_train_test_ids(collection, target, train_percentage=0.8, seed=None):
    """get random train/test split, keeping the proportion of pos/neg
       classes the same"""
    i = 0
    if seed:
        random.seed(seed)
    pos_train_ids = []
    neg_train_ids = []
    pos_test_ids = []
    neg_test_ids = []
    pos_docs = collection.find({'target': target})
    neg_docs = collection.find({'target': {'$ne': target}})
    for doc in pos_docs:
        if random.random() < train_percentage:
            pos_train_ids.append(doc['_id'])
        else:
            pos_test_ids.append(doc['_id'])
    for doc in neg_docs:
        if random.random() < train_percentage:
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
    zipped_train = list(zip(X_train_ids, y_train))
    zipped_test = list(zip(X_test_ids, y_test))
    random.shuffle(zipped_train)
    random.shuffle(zipped_test)
    X_train_ids, y_train = zip(*zipped_train)
    X_test_ids, y_test = zip(*zipped_test)
    return i, X_train_ids, X_test_ids, np.array(y_train), np.array(y_test)
    
def _get_k_fold_ids(collection, target, seed=None, k_folds=5):
    """generate k_fold indices for X_train, X_test, y_train, y_test"""
    if k_folds == 1:
        return _get_train_test_ids(collection, target, train_percentage=0.8)
    if k_folds < 2:
        raise ValueError('Minimum k_folds is 2')
    if seed:
        random.seed(seed)
    pos_documents = []
    neg_documents = []
    pos_docs = collection.find({'target': target})
    neg_docs = collection.find({'target': {'$ne': target}})
    for doc in pos_docs:
        pos_documents.append(doc['_id'])
    for doc in neg_docs:
        neg_documents.append(doc['_id'])
    random.shuffle(pos_documents)
    random.shuffle(neg_documents)
    pos_count = len(pos_documents) // k_folds
    neg_count = len(neg_documents) // k_folds
    for fold in range(k_folds-1):
        X_pos_test = pos_documents[(fold*pos_count):((fold*pos_count)+pos_count)]
        X_neg_test = neg_documents[(fold*neg_count):((fold*neg_count)+neg_count)]
        X_pos_train = set(pos_documents)-set(X_pos_test)
        X_neg_train = set(neg_documents)-set(X_neg_test)
        pos_train_y_list = list(np.ones(len(X_pos_train)))
        neg_train_y_list = list(np.zeros(len(X_neg_train)))
        pos_test_y_list = list(np.ones(len(X_pos_test)))
        neg_test_y_list = list(np.zeros(len(X_neg_test)))
        X_train_ids = list(X_pos_train) + list(X_neg_train)
        X_test_ids = list(X_pos_test) + list(X_neg_test)
        y_train = pos_train_y_list + neg_train_y_list
        y_test = pos_test_y_list + neg_test_y_list  
        yield fold, X_train_ids, X_test_ids, y_train, y_test
    X_pos_test = pos_documents[(k_folds-1)*pos_count:]
    X_neg_test = neg_documents[(k_folds-1)*neg_count:]
    X_pos_train = pos_documents[:k_folds*pos_count]
    X_neg_train = neg_documents[:k_folds*neg_count]
    pos_train_y_list = list(np.ones(len(X_pos_train)))
    neg_train_y_list = list(np.zeros(len(X_neg_train)))
    pos_test_y_list = list(np.ones(len(X_pos_test)))
    neg_test_y_list = list(np.zeros(len(X_neg_test)))
    X_train_ids = X_pos_train + X_neg_train
    X_test_ids = X_pos_test + X_neg_test
    y_train = pos_train_y_list + neg_train_y_list
    y_test = pos_test_y_list + neg_test_y_list  
    yield k_folds-1, X_train_ids, X_test_ids, y_train, y_test 

    

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