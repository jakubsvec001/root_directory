import scipy
import pickle
import src.wiki_finder as wf
import src.page_disector as disector
import src.model as m
from pymongo import MongoClient
from gensim import corpora, models


def deploy_model(file, target, n_grams, feature_length=100000, limit=None):
    """deploy the model"""
    mc = MongoClient()
    db = mc['wiki_cache']
    collection = db[f'{target}_logistic_predictions']
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
        model = pickle.load(open('nlp_training_data/final_logistic_model.pkl',
                                 'rb'))
        print('Logistic Regression Model Loaded!')
    except ValueError:
        print('Could not find Logistic Regression Model at' +
              f'nlp_training_data/final_logistic_model.pkl')
    line_gen = wf.get_lines_bz2(file)
    page_gen = wf.page_generator(line_gen, limit=limit)
    results = []
    for raw_xml in page_gen:
        results = wf.identify_page(raw_xml)
        title = results['title']
        xml = results['full_raw_xml']
        results = disector.disect_page(title, xml)
        parsed_xml = results['feature_union']
        n_gram_article = m._stem_and_ngramizer(parsed_xml, n_grams)
        article_bow = dictionary.doc2bow(n_gram_article)
        article_tfidf = tfidf[article_bow]
        if len(article_tfidf) == 0:
            save_to_db(collection, title, terms=[], prediction=0,
                       text=parsed_xml)
            continue
        column, value = list(zip(*article_tfidf))
        row = [0] * len(column)
        scipy_sparse_row = scipy.sparse.csr_matrix((value,
                                                   (row, column)),
                                                   shape=(1, feature_length))
        prediction = model.predict_proba(scipy_sparse_row)
        words = [dictionary[item] for item in column]
        save_to_db(collection, title, terms=list(zip(words, value)),
                   prediction=prediction[0][1], text=parsed_xml)


def save_to_db(collection, title, terms, prediction, text):
    ping = collection.find_one({'title': title})
    if ping is None:
        document = {'title': title, 'terms': terms, 'prediction': prediction,
                    'text': text}
        collection.insert_one(document)
