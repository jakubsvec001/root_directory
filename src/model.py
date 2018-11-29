import src.wiki_db_parser as wdbp 
import sys
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from bson.objectid import ObjectId
from gensim import corpora, models
from gensim.parsing.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from smart_open import smart_open
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
import numpy as np  
from sklearn.feature_extraction import stop_words


def save_txt_nlp_train_data(db_name, collection_name, target, fileout_dir, subset):
    """use a mongodb collection and a subsampled target subset percentage
    to create a .txt file with one line per document"""
    mc = MongoClient()
    db = mc[db_name]
    col = db[collection_name]
    target_pages = col.find({'target':target})
    df = pd.DataFrame(list(target_pages))['feature_union']
    subsampled_df = df.sample(frac=subset, replace=False)
    with open(fileout_dir, 'w') as fout:
        for row in subsampled_df:
            if row != 'nan':
                fout.write(row + '\n')


def train_save_dictionary_corpus(filein, n_grams, target):
    """Use gensim to create a streamed dictionary"""
    dictionary = corpora.Dictionary(_list_grams(filein, n_grams))
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n = 100000)
    dictionary.save(f'nlp_training_data/{target}_dictionary.dict')
    corpus = [dictionary.doc2bow(word) for word in _list_grams(filein, n_grams)]
    corpora.MmCorpus.serialize(f'nlp_training_data/{target}_corpus.mm', corpus)
    return dictionary, corpus


def train_save_tfidf(filein, target):
    """input is a bow corpus saved as a tfidf file. The output is 
       a saved tfidf corpus"""
    try:
        corpus = corpora.MmCorpus(filein)
    except:
        raise NameError('HRMMPH. The file does not seem to exist. Create a file '+
                        'first by running the "train_save_dictionary_corpus" function.')
    tfidf = models.TfidfModel(corpus)
    tfidf.save(f'../nlp_training_data/{target}_tfidf_model.tfidf')
    tfidf_corpus = tfidf[corpus]
    return tfidf_corpus


def train_multinomial_nb(db_name, collection_name, num_target_col):
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    docs = collection.find()
    items = []
    for doc in docs:
        items.append(doc[num_target_col])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    items = np.array(items)
    return items
    sss.get_n_splits(items.T[0], items.T[1])


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