import pandas as pd
import numpy as np

def clean_sparqle(infile, outfile)
    data = pd.read_csv(infile)
    wikidata_url_base = 'http://www.wikidata.org/entity/'
    url_length = len(wikidata_url_base)
    data['wikidata_id'] = data['item']
    data.loc[:, 'wikidata_id'] = data['wikidata_id'].apply(lambda x: x[url_length : ])
    data = data[['wikidata_id', 'sitelink', 'article', 'item']]
    data.columns = ['wikidata_id', 'article_name', 'wikipedia_url', 'wikidata_url']
    data.to_csv('data/math/math_clean_sparql_786', sep='\t', encoding='utf-8')