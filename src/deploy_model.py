import scipy
import sys
import pickle
import src.wiki_finder as wf
import src.page_disector as disector
import src.model as m
from pymongo import MongoClient
from gensim import corpora, models


def deploy_model(file, target, n_grams, feature_count=100000, limit=None):
    """deploy a logistic model

        Parameters
        ----------
        file, target, n_grams, feature_count=100000, limit=None

        Returns
        -------
        None
    """
    mc = MongoClient()
    db = mc['wiki_cache']
    collection = db[f'{target}_logistic_predictions_3']
    try:
        dictionary = corpora.Dictionary.load(
            f'nlp_training_data/{target}_full.dict')
        print('Dictionary Loaded!')
    except ValueError:
        print('Could not find dictionary at ' +
              f'nlp_training_data/{target}_full.dict')
    try:
        tfidf = models.TfidfModel.load(
            f'nlp_training_data/{target}_full.tfidf')
        print('Tfidf model loaded!')
    except ValueError:
        print('Could not find tfidf model at ' +
              f'nlp_training_data/{target}_full.tfidf')
    try:
        model = pickle.load(open('nlp_training_data/' +
                                 f'{target}_final_logistic_model.pkl',
                                 'rb'))
        print('Logistic Regression Model Loaded!')
    except ValueError:
        print('Could not find Logistic Regression Model at' +
              f'nlp_training_data/{target}_full_logistic_model.pkl')
    line_gen = wf.get_lines_bz2(file)
    page_gen = wf.page_generator(line_gen, limit=limit)
    results = []
    saved = 0
    searched = 0
    for raw_xml in page_gen:
        searched += 1
        results = wf.identify_page(raw_xml)
        title = results['title']
        xml = results['full_raw_xml']
        results = disector.disect_page(title, xml)
        parsed_xml = results['feature_union']
        n_gram_article = m._stem_and_ngramizer(parsed_xml, n_grams)
        article_bow = dictionary.doc2bow(n_gram_article)
        article_tfidf = tfidf[article_bow]
        if len(article_tfidf) == 0:
            save_to_db(collection, title, prediction=0)
            continue
        column, value = list(zip(*article_tfidf))
        row = [0] * len(column)
        scipy_sparse_row = scipy.sparse.csr_matrix((value,
                                                   (row, column)),
                                                   shape=(1, feature_count))
        prediction = model.predict_proba(scipy_sparse_row)
        if prediction[0][1] >= 0.18:
            saved += 1
            save_to_db(collection, title, prediction=prediction[0][1])
            sys.stdout.write('\r' + f'Searched: {searched}, Saved: {saved}')


def save_to_db(collection, title, prediction):
    """save title and prediction to database.
        ----------
        Parameters
        ----------
        collection, title, prediction

        Returns
        -------
        None
    """
    ping = collection.find_one({'title': title})
    if ping is None:
        document = {'title': title, 'prediction': prediction}
        collection.insert_one(document)
