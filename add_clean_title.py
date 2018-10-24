import pandas as pd
import numpy as np

def add_clean_title(infile):
    data = pd.read_csv(str(infile), sep='\t')
    data['title_clean'] = data['title'].apply(lambda x: x.replace('_', ' '))
    data.to_csv(str(infile), sep='\t', encoding='utf-8', index=False)