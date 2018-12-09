import src.wiki_db_parser as wdbp
import src.page_disector as disector
import src.wiki_finder as wf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import sys
import pickle
import scipy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (log_loss, confusion_matrix,
                             roc_curve, auc, precision_score,
                             recall_score)
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
import pandas as pd


def cross_validate_multinomial_nb(db_name,
                                  collection_name,
                                  target,
                                  n_grams=3,
                                  shuffle=True,
                                  feature_count=100000,
                                  build_sparse_matrices=True):
    """train naive bayes model using a train/test
    split. Return predictions, score
        ----------
        Parameters
        ----------
        str: db_name,
        str: collection_name,
        str: target,
        int: n_grams=3,
        bool: shuffle=True,
        int: feature_count=100000,
        bool: build_sparse_matrices=True
        
        Returns
        -------
        list: y_test,
        list: preds,
        float: score,
        sklearn model: model
    """
    start = default_timer()
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    print('Generating stratified train/test split ids from dataset')
    output = _get_train_test_ids(collection, target, shuffle=shuffle,
                                 train_percentage=0.8, seed=1)
    _, X_train_ids, X_test_ids, y_train, y_test, X_pos_train = output
    if build_sparse_matrices:
        output = _build_matrices(start, db_name, collection_name,
                                 target, n_grams, collection,
                                 feature_count, X_train_ids, X_test_ids,
                                 pos_ids=X_pos_train, training=False)
        scipy_X_train, scipy_X_test = output
    else:
        try:
            scipy_X_train = pickle.load(open(
                f'nlp_training_data/{target}_X_train_tfidf.pkl', 'rb'))
            scipy_X_test = pickle.load(open(
                f'nlp_training_data/{target}_X_test_tfidf.pkl', 'rb'))
            print('Loaded saved dictionary and tfidf models')
        except ValueError:
            print(f"Can't find saved dictionary. Try " +
                  "running function with: build_dict_tfidf=True")
    model = MultinomialNB().fit(scipy_X_train, y_train)
    preds = model.predict_proba(scipy_X_test)
    end = default_timer()
    print(f'    Elapsed time: {round((end-start)/60, 2)} minutes')
    print('SCORING model...')
    score = log_loss(y_test, preds.T[1])
    print('DONE!')
    _plot_roc_curves('Multinomial_NM', target, [y_test],
                     [preds], None, feature_count)
    return y_test, preds, score, model


def logistic_regression_cv(db_name, collection_name, target,
                           Cs, shuffle=True, feature_count=100000,
                           n_grams=3, build_sparse_matrices=False):
    """gridsearch without cross validation
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    start = default_timer()
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    print('Generating stratified train/test split ids from dataset')
    output = _get_train_test_ids(collection, target, shuffle=shuffle,
                                 train_percentage=0.8, seed=1)
    _, X_train_ids, X_test_ids, y_train, y_test, X_pos_train = output
    if build_sparse_matrices:
        output = _build_matrices(start, db_name, collection_name,
                                 target, n_grams, collection, feature_count,
                                 X_train_ids, X_test_ids, pos_ids=X_pos_train,
                                 training=True)
        scipy_X_train, scipy_X_test = output
    else:
        try:
            scipy_X_train = pickle.load(open(
                f'nlp_training_data/{target}_scipy_X_train.pkl', 'rb'))
            scipy_X_test = pickle.load(open(
                f'nlp_training_data/{target}_scipy_X_test.pkl', 'rb'))
            print('Loaded saved scipy_X_train and scipy_X_test matrices')
        except ValueError:
            print(f"Can't find saved sparse matrices")
    model_list = []
    pred_list = []
    score_list = []
    for c in Cs:
        print(f'RUNNING GRIDSEARCH FOR {c}...')
        print('    FITTING logistic regression model...')
        model = LogisticRegressionCV(penalty='l2',
                                     solver='saga',
                                     Cs=[c],
                                     scoring='neg_log_loss',
                                     cv=5,
                                     verbose=0,
                                     n_jobs=2)
        model.fit(scipy_X_train, y_train)
        print('    GENERATING predictions...')
        predictions = model.predict_proba(scipy_X_test)
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print('    SCORING model...')
        score = log_loss(y_test, predictions.T[1])
        print('    DONE!')
        model_list.append(model)
        pred_list.append(predictions)
        score_list.append(score)
        end = default_timer()
        print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
        print(f'score: {score}')
        print()
    _plot_roc_curves('Logistic Regression', target, y_test, Cs,
                     pred_list, feature_count)
    best_score_idx = np.argmin(score_list)
    best_score = score_list[best_score_idx]
    best_model = model_list[best_score_idx]
    best_predictions = pred_list[best_score_idx]
    # save best model
    pickle.dump(
        best_model,
        open(f'nlp_training_data/{target}_best_logistic_reg_cv_model.pkl',
             'wb'))
    return (best_score, best_model, best_predictions,
            y_test, X_test_ids, scipy_X_test)


def logistic_regression_model(db_name, collection_name,
                              target, C, feature_count=100000,
                              n_grams=3, build_sparse_matrices=False):
    """Build, return, save best model for deployment on all data
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    output = _get_train_test_ids(collection, target, shuffle=False,
                                 train_percentage=1, seed=1)
    _, X_train_ids, _, y_train, _, _ = output
    C = float(C)
    if build_sparse_matrices:
        # save target article content to text file
        _save_txt_nlp_data(db_name, collection_name, target,
                           X_train_ids, training=False)
        # create dictionary from target text file
        dictionary, _ = _train_save_dictionary_corpus(
                            f'nlp_training_data/{target}_full.txt',
                            n_grams, target, training=False,
                            feature_count=feature_count)
        # generate tfidf model
        tfidf = _train_save_tfidf(f'nlp_training_data/{target}_full_corpus.mm',
                                  target, training=False)
        # get ids of target
        _make_temporary_txt(collection, X_train_ids)
        # create tfidf matrix from target articles
        train_bow = [dictionary.doc2bow(word) for word in _list_grams(
                        '/tmp/docs_for_sparse_vectorization.txt',
                        n_grams=n_grams)]
        X_train_tfidf = tfidf[train_bow]
        print('Generating and saving scipy_X_train...')
        scipy_X_train = matutils.corpus2csc(X_train_tfidf).transpose()
        # save sparse matrix
        pickle.dump(scipy_X_train, open(
            'nlp_training_data/{target}_final_scipy_sparse_matrix.pkl', 'wb'))
    else:
        scipy_X_train = pickle.load(open(
            f'nlp_training_data/{target}_final_scipy_sparse_matrix.pkl', 'rb'))
    print('Training Logisitic Regression on full training dataset')
    model = LogisticRegression(penalty='l2', solver='saga', C=C)
    model.fit(scipy_X_train, y_train)
    print('Saving model')
    pickle.dump(model, open(
        f'nlp_training_data/{target}_final_logistic_model.pkl', 'wb'))
    return model


def generate_confusion_matrix(y_test, predictions, start=10, stop=90, steps=5):
    """generate confusion matrices at various thresholds
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    matrices = []
    for i in range(start, stop, steps):
        matrix = confusion_matrix(y_test, predictions[:, 1] > i/100)
        matrices.append(matrix)
        print(i, matrix)
    return steps, matrices


def generate_precision_recall_scores(y_test, predictions, threshold):
    """
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    threshold = float(threshold)
    precision = precision_score(y_test, predictions[:, 1] > threshold)
    recall = recall_score(y_test, predictions[:, 1] > threshold)
    return precision, recall


def generate_feature_importance_graph(target, word_import):
    """save a horizontal graph of the top most important features
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    target_string = target.title().replace('_', ' ')
    title = f'{target_string} Feature Importance'
    print(title)
    df = pd.DataFrame(word_import, columns=['word', 'importance'])
    middle = df.shape[0] // 2
    df_low = df.iloc[:5, :]
    df_mid = df.iloc[middle-5:middle+5, :]
    df_high = df.iloc[-5:, :]
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.barh(y=df_low['word'], width=df_low['importance'])
    ax.barh(y=df_mid['word'], width=df_mid['importance'])
    ax.barh(y=df_high['word'], width=df_high['importance'])
    ax.set_title(title)
    plt.savefig(f'images/{target}_word_import.png')
    plt.show()
    return df


def get_confusion_titles(model, target, y_test, prediction, threshold,
                         X_test_ids):
    """return and save titles in different buckets or predictions
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    threshold = float(threshold)
    try:
        scipy_X_test = pickle.load(
            open(f'nlp_training_data/{target}_scipy_X_test.pkl', 'rb'))
        print('Loaded saved scipy_X_test matrices')
    except ValueError:
        print(f"Can't find saved sparse matrices")
    collection = MongoClient()['wiki_cache']['all']
    predicted = model.predict_proba(scipy_X_test)
    df = pd.DataFrame(y_test, columns=['actual'])
    df['predicted'] = predicted[:, 1]
    df['threshold_pred'] = df['predicted'] > threshold
    df['_id'] = X_test_ids
    titles = []
    for item in X_test_ids:
        article = collection.find_one({'_id': ObjectId(item)})
        titles.append(article['title'])
    df['title'] = titles
    df['FP'] = np.where((df['actual'] == False) &
                        (df['threshold_pred'] == True), True, False)
    df['TP'] = np.where((df['actual'] == True) &
                        (df['threshold_pred'] == True), True, False)
    df['FN'] = np.where((df['actual'] == True) &
                        (df['threshold_pred'] == False), True, False)
    df['TN'] = np.where((df['actual'] == False) &
                        (df['threshold_pred'] == False), True, False)
    FP = df[df['FP'] == True][['title', 'predicted',
                               'actual', 'threshold_pred']]
    TN = df[df['TN'] == True][['title', 'predicted',
                               'actual', 'threshold_pred']]
    FP = df[df['FP'] == True][['title', 'predicted',
                               'actual', 'threshold_pred']]
    FN = df[df['FN'] == True][['title', 'predicted',
                               'actual', 'threshold_pred']]
    FP.to_csv(f'results/FP_{target}_{threshold}_confusion_results.csv',
              sep='\t', index=False)
    TN.to_csv(f'results/TN_{target}_{threshold}_confusion_results.csv',
              sep='\t', index=False)
    FP.to_csv(f'results/FP_{target}_{threshold}_confusion_results.csv',
              sep='\t', index=False)
    FN.to_csv(f'results/FN_{target}_{threshold}_confusion_results.csv',
              sep='\t', index=False)
    return FP, TN, FP, FN


def _build_matrices(start, db_name, collection_name,
                    target, n_grams, collection,
                    feature_count, X_train_ids,
                    X_test_ids, pos_ids, training=True):
    """builds, saves, and returns scipy sparse matrices
    for training and testing sklearn models
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    _save_txt_nlp_data(db_name, collection_name, target, pos_ids, training)
    dictionary, _ = _train_save_dictionary_corpus(
        f'nlp_training_data/{target}_subset.txt', n_grams, target,
        training=training, feature_count=feature_count)
    tfidf = _train_save_tfidf(f'nlp_training_data/{target}_subset_corpus.mm',
                              target, training=training)
    print('    CREATING temporary txt file...')
    _make_temporary_txt(collection, X_train_ids)
    end = default_timer()
    print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
    print('    CREATING training set bow with txt file...')
    train_bow = [dictionary.doc2bow(word) for word in _list_grams(
                '/tmp/docs_for_sparse_vectorization.txt', n_grams=n_grams)]
    end = default_timer()
    print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
    print('    CREATING training set tfidf with txt file...')
    X_train_tfidf = tfidf[train_bow]
    pickle.dump(X_train_tfidf, open(
        f'nlp_training_data/{target}_X_train_tfidf.pkl', 'wb'))
    end = default_timer()
    print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
    print('    CONVERTING tfidf training model to scipy sparse matrix...')
    scipy_X_train = matutils.corpus2csc(X_train_tfidf).transpose()
    pickle.dump(scipy_X_train,
                open(f'nlp_training_data/{target}_scipy_X_train.pkl',
                     'wb'))
    end = default_timer()
    print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
    _make_temporary_txt(collection, X_test_ids)
    print('    CREATING test set bow with txt file...')
    test_bow = [dictionary.doc2bow(word) for word in
                _list_grams('/tmp/docs_for_sparse_vectorization.txt',
                n_grams=n_grams)]
    print('    CREATING test tfidf...')
    X_test_tfidf = tfidf[test_bow]
    end = default_timer()
    print(f'        Elapsed time: {round((end-start)/60, 2)} minutes')
    print('    CONVERTING tfidf testing model to scipy sparse matrix...')
    scipy_X_test = matutils.corpus2csc(X_test_tfidf).transpose()
    pickle.dump(scipy_X_test, open(
                f'nlp_training_data/{target}_scipy_X_test.pkl', 'wb'))
    return scipy_X_train, scipy_X_test


def _plot_roc_curves(model_type, target, y_test, Cs, pred_list, feature_count):
    """plot roc curve for each cross_validated model
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(1, 1, figsize=((10, 10)))
    for i in range(len(pred_list)):
        fpr, tpr, thresholds = roc_curve(y_test, pred_list[i][:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label=f'ROC C: {Cs[i]} (AUC = {roc_auc:0.2f})')
        print()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{target} {model_type} ROC Feature Count = {feature_count}')
    ax.legend(loc="lower right")
    fig.savefig('images/' +
                f'{target}_roc_cv_logistic_regression_{feature_count}.png')
    print(f'saved roc curves to ' +
          f'images/{target}_roc_cv_logistic_regression_{feature_count}.png\n')
    fig.show()


def _save_txt_nlp_data(db_name, collection_name, target,
                       pos_ids=None, training=True):
    """use a mongodb collection and a subsampled target subset percentage
    to create a .txt file with one line per document
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    print('Making txt file of subset of target class')
    mc = MongoClient()
    db = mc[db_name]
    col = db[collection_name]
    target_pages = col.find({'target': target})
    df = pd.DataFrame(list(target_pages))[['_id', 'feature_union']]
    training_df = df[df['_id'].isin(pos_ids)]['feature_union']
    if training:
        with open(f'nlp_training_data/{target}_subset.txt', 'w') as fout:
            for row in training_df:
                if row != 'nan':
                    fout.write(row + '\n')
    else:
        with open(f'nlp_training_data/{target}_full.txt', 'w') as fout:
            for row in df['feature_union']:
                if row != 'nan':
                    fout.write(row + '\n')
    print('DONE!')


def _train_save_dictionary_corpus(filein, n_grams, target,
                                  training=True,
                                  feature_count=100000):
    """Use gensim to create a streamed dictionary.
    filein is the file used to train the dictionary and tfidf
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    print('Building dictionary...')
    if training:
        dictionary = corpora.Dictionary(_list_grams(filein, n_grams))
        print('Dictionary len before filter = ', len(dictionary))
        dictionary.filter_extremes(no_below=5, no_above=0.5,
                                   keep_n=feature_count)
        print('Dictionary len after filter = ', len(dictionary))
        dictionary.save(
            f'nlp_training_data/{target}_subset.dict')
        corpus = [dictionary.doc2bow(word) for word in _list_grams(
            filein, n_grams)]
        corpora.MmCorpus.serialize(
            f'nlp_training_data/{target}_subset_corpus.mm', corpus)
        print(f'saved nlp_training_data/{target}_subset_corpus.mm')
    else:
        dictionary = corpora.Dictionary(_list_grams(filein, n_grams))
        print('Dictionary len before filter = ', len(dictionary))
        dictionary.filter_extremes(no_below=5, no_above=0.5,
                                   keep_n=feature_count)
        print('Dictionary len after filter = ', len(dictionary))
        dictionary.save(f'nlp_training_data/{target}_full.dict')
        corpus = [dictionary.doc2bow(word) for word in _list_grams(
            filein, n_grams)]
        corpora.MmCorpus.serialize(
            f'nlp_training_data/{target}_full_corpus.mm', corpus)
    print('DONE!')
    return dictionary, corpus


def _train_save_tfidf(filein, target, training=True):
    """input is a bow corpus saved as a tfidf file. The output is
       a saved tfidf corpus
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    print('Building TFIDF model')
    if training:
        try:
            corpus = corpora.MmCorpus(filein)
        except ValueError:
            raise NameError('HRMMPH. The file does not seem to exist. ' +
                            'Create a file first by running the ' +
                            '"train_save_dictionary_corpus" function.')
        tfidf = models.TfidfModel(corpus)
        tfidf.save(f'nlp_training_data/{target}_subset.tfidf')
    else:
        try:
            corpus = corpora.MmCorpus(filein)
        except ValueError:
            raise NameError('HRMMPH. The file does not seem to exist. ' +
                            'Create a file first by running the ' +
                            '"train_save_dictionary_corpus" function.')
        tfidf = models.TfidfModel(corpus)
        tfidf.save(f'nlp_training_data/{target}_full.tfidf')
    print('DONE!')
    return tfidf


def _make_temporary_txt(collection, ids):
    """make a temporary txt file of documents
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    with open('/tmp/docs_for_sparse_vectorization.txt', 'w') as fout:
        for id in ids:
            res = collection.find_one(id)
            text = res['feature_union']
            fout.write(text + '\n')


def _get_train_test_ids(collection, target, train_percentage=0.8,
                        seed=None, shuffle=False):
    """get random train/test split, keeping the proportion of pos/neg
       classes the same
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    i = 0
    if seed:
        random.seed(seed)
    pos_documents = []
    neg_documents = []
    if shuffle:
        random.shuffle(pos_documents)
        random.shuffle(neg_documents)
    pos_docs = collection.find({'target': target})
    neg_docs = collection.find({'target': {'$ne': target}})
    for doc in pos_docs:
        pos_documents.append(doc['_id'])
    for doc in neg_docs:
        neg_documents.append(doc['_id'])
    pos_train_len = int(len(pos_documents) * train_percentage)
    neg_train_len = int(len(neg_documents) * train_percentage)
    X_pos_train = pos_documents[:pos_train_len]
    X_pos_test = pos_documents[pos_train_len:]
    X_neg_train = neg_documents[:neg_train_len]
    X_neg_test = neg_documents[neg_train_len:]
    pos_train_y_list = list(np.ones(len(X_pos_train)))
    neg_train_y_list = list(np.zeros(len(X_neg_train)))
    pos_test_y_list = list(np.ones(len(X_pos_test)))
    neg_test_y_list = list(np.zeros(len(X_neg_test)))
    X_train_ids = X_pos_train + X_neg_train
    X_test_ids = X_pos_test + X_neg_test
    y_train = pos_train_y_list + neg_train_y_list
    y_test = pos_test_y_list + neg_test_y_list
    if shuffle:
        zipped_train = list(zip(X_train_ids, y_train))
        zipped_test = list(zip(X_test_ids, y_test))
        random.shuffle(zipped_train)
        random.shuffle(zipped_test)
        X_train_ids, y_train = zip(*zipped_train)
        X_test_ids, y_test = zip(*zipped_test)
    return (i, X_train_ids, X_test_ids, np.array(y_train),
            np.array(y_test), X_pos_train)


def _get_k_fold_ids(collection, target, seed=None, k_folds=5):
    """generate k_fold indices for X_train, X_test, y_train, y_test
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
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
        X_pos_test = pos_documents[(fold*pos_count):((
            fold*pos_count)+pos_count)]
        X_neg_test = neg_documents[(fold*neg_count):((
            fold*neg_count)+neg_count)]
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
    """DEPRICATED! remove words that appear only once
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    once_ids = [tokenid for tokenid, docfreq in
                dictionary.dfs.items() if docfreq <= 2]
    dictionary.filter_tokens(once_ids)
    return dictionary


def _list_grams(filein, n_grams):
    """for each document, yield a list of strings
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    with open(filein, 'r') as fin:
        for line in fin:
            yield _stem_and_ngramizer(line, n_grams)


def _stem_and_ngramizer(line, n_grams):
    """stem all the words, generate ngrams, and return
       a list of all stemmed words and ngram phrases
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    p = PorterStemmer()
    s = SnowballStemmer('english')
    stopped = [word for word in line.split() if
               word not in stop_words.ENGLISH_STOP_WORDS]
    stems = [s.stem(word) for word in stopped]
    grams = [[' '.join(stems[i:i+n]) for i in
              range(len(stems)-n+1)] for n in range(1, n_grams + 1)]
    return [item for sublist in grams for item in sublist]


def main(arg1, arg2):
    """
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """
    pass


if __name__ == '__main__':
    if sys.argv != 3:
        raise NameError('Incorrect parameter count. Use: ' +
                        'python model.py arg1 arg2')
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    main(arg1, arg2)
