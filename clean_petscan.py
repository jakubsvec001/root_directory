import pandas as pd
import numpy as np

def clean_petscan(infile, outfile):
    petscan = pd.read_csv(str(infile))
    url_base = 'https://en.wikipedia.org/wiki/'
    petscan.drop(['number', 'namespace', 'touched', 'length', 'pageid'], axis=1, inplace=True)
    petscan['url'] = url_base + petscan['title']
    petscan = petscan[['wikidata', 'title', 'url']]
    petscan.to_csv(str(outfile), sep='\t', encoding='utf-8', index=False)