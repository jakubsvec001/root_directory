import pandas as pd
import numpy as np

def clean_csv_list(file):
    """input a csv file from wikiscarper output"""
    data = pd.read_csv(file, sep='\t', encoding='utf-8')
    data['cleaned_url'] = data['url'].apply(internal_link_finder)
    data = data[['url', 'cleaned_url', 'depth']]
    data.to_csv(file, sep='\t', encoding='utf-8', index=False)

def internal_link_finder(row):
    if row.startswith('/wiki/'):
        return row[len('/wiki/'):].replace('_', ' ')    
    else:
        return np.nan

def limit_depth(file, d):
    """take in a wikiscraper output file and limit its depth. 
    Depth is indexed starting at zero"""
    data = pd.read_csv(file, sep='\t', encoding='utf-8')
    d_max = data.loc[:,'depth'].max()
    if d-1 >= d_max:
        raise ValueError('WEEEE ERRRR! Selected depth exceeds depth of input data. ' 
        f'Max depth of input data is {d_max} and has {d_max+1} levels')
    new_data = data[data['depth']<d]
    directory = file.partition('/')[0]
    filename = file.partition('/')[-1]
    category = filename.partition('_')[0]
    new_data.to_csv(f'{directory}/{category}_d{d}.csv', sep='\t', encoding='utf-8', index=False)
