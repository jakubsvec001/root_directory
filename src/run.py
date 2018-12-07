import src.deploy_model as dm
import src.model as m
import src.wiki_finder as wf
import src.page_disector as disector
import src.wiki_list_cleaner as wlc


def get_depth_csv(category, depth):
    """check if a csv of proper depth exists,
    if not, create a new """
    pass


def get_category():
    """get input from user about category of choice"""
    c = ['aeronautics', 'arts', 'biology', 'chemistry', 'computer science',
         'engineering', 'mathematics', 'philosophy', 'physics']
    category = input("""What category are you interested finding?\n\n
        Available categories: \n
          - Aeronautics\n
          - Arts\n
          - Biology\n
          - Chemistry\n
          - Computer Science\n
          - Engineering\n
          - Mathematics\n
          - Philosophy\n
          - Physics\n\n
Category: """)
    if category.lower() in c:
        return category.lower().replace(' ', '_')
    else:
        print('Please enter a valid category')
        get_category()


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


def main():
    """"""
    category = get_category()
    depth = get_depth()
    wlc.limit_depth(f'seed/{category}_d5.csv', depth)
    result = m.logistic_regression_cv(db_name='wiki_cache',
                                      collection_name='all',
                                      target=category,
                                      Cs=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                                      feature_count=100000,
                                      build_sparse_matrices=True)
    best_score, best_model, best_predictions, y_test, X_test_ids = result

