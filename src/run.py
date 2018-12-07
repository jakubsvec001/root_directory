import src.deploy_model as deploy
import src.model as m
import src.wiki_finder as wf
import src.page_disector as disector
import src.wiki_list_cleaner as wlc
from gensim import corpora, models
import pandas as pd
import numpy as np
import glob
import sys


def get_depth_csv(target, depth):
    """check if a csv of proper depth exists,
    if not, create a new """
    pass


def get_target():
    """get input from user about category of choice"""
    c = ['aeronautics', 'arts', 'biology', 'chemistry', 'computer science',
         'engineering', 'mathematics', 'philosophy', 'physics']
    target = input("""Which category are you interested building?\n\n
        Available categories: \n
          - Aeronautics\n
          - Arts\n
          - Biology\n
          - Chemistry\n
          - Computer Science\n
          - Engineering\n
          - Mathematics\n
          - Philosophy\n
          - Physics\n
Category: """)
    if target == 'q':
        return target
    if target.lower() in c:
        return target.lower().replace(' ', '_')
    else:
        print('Please enter a valid category')
        get_target()


def get_depth():
    """get depth from user"""
    depth = input("""How deep in the tree would you like to go?
    (2 or 3)\n\n
    Depth: """)
    if depth.isdigit() is False:
        print('Please enter a number.\n')
        get_depth()
    elif depth != '2' and depth != '3':
        print('Please select 2 or 3.\n')
        get_depth()
    else:
        return int(depth)


def train_final_model_input():
    print('Would you like to continue and train the final model? ')
    response = input('Y / N: ')
    if response.lower() != 'y' and response.lower() != 'n':
        train_final_model_input()
    return response.lower()


# [0.1, 0.12, 0.15, 0.17, 0.2, 0.22,
# 0.25, 0.27, 0.3, 0.32, 0.35, 0.4,
# 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1],


def main(from_scratch=False):
    """Cross validate a model, train a final model"""
    while True:
        target = get_target()
        if target == 'q':
            print('Quitting...')
            break
        depth = 3
        wlc.limit_depth(f'seed/{target}_d5.csv', depth)
        result = m.logistic_regression_cv(db_name='wiki_cache',
                                          collection_name='all',
                                          target=target,
                                          Cs=[5, 10],
                                          feature_count=100000,
                                          build_sparse_matrices=from_scratch)
        (best_score, best_model, best_predictions,
         y_test, X_test_ids, scipy_X_test) = result
        dictionary = corpora.Dictionary.load(
            f'nlp_training_data/{target}_subset.dict')
        print("Best model:")
        print(best_model)
        print("Best score:")
        print(best_score)
        print("Confusion matrices: ")
        confusion_matrices = m.generate_confusion_matrix(
            y_test, best_predictions)
        threshold = input('Please select a threshold from ' +
                          'the confusion matrices:\n\nThreshold: ')
        FP, TN, FP, FN = m.get_confusion_titles(
            best_model, target, y_test,
            best_predictions, threshold,
            X_test_ids)
        print(f'Saved confusion titles to results folder')
        print('Sample of False Negative titles: ')
        print(FN.head(20))
        print('Sample of False Positive titles: ')
        print(FP.head(20))
        precision, recall = m.generate_precision_recall_scores(
            y_test,
            best_predictions,
            threshold)
        print('Precision score:')
        print(precision)
        print('Recall score:')
        print(recall)
        sort_coefs = np.sort(best_model.coef_)
        argsort_coefs = np.argsort(best_model.coef_)
        terms = []
        for item in argsort_coefs[0]:
            terms.append(dictionary[item])
        feature_importances = list(zip(terms, sort_coefs[0]))
        return best_model, dictionary, feature_importances
        response = train_final_model_input()
        if response == 'n':
            break
        best_c = best_model.C_
        df_coef = pd.DataFrame(best_model.coef_)
        m.logistic_regression_model('wiki_cache', 'all', target=target,
                                    C=best_c, build_sparse_matrices=True)
        return best_model, df_coef