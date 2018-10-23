import pandas as pd
import numpy as np

def merge_sparql_petscan(sparql_in, petscan_in, outfile):
    sparql = pd.read_csv(sparql_in, sep='\t')
    sparql = sparql[['wikidata_id', 'article_name', 'wikipedia_url']]
    sparql.columns = ['wikidata', 'title', 'url']
    petscan = pd.read_csv(petscan_in, sep='\t')
    data = pd.concat([petscan, sparql], axis=0)
    data = data[~data['wikidata'].duplicated()]
    length = data.shape[0]
    data.to_csv(str(outfile)+'_'+ str(length), sep='\t', encoding='utf-8')