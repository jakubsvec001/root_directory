import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import urllib.request
import json
import csv
import sys
import time


class WikidataConverter():
    """cleans a csv file of wikidata nodes and finds 
       english wikipedia articles, if available.
    """
    def __init__(self):
        self.total_written = 0
        self.total_skipped = 0
    
    def parse(self, input_file, output_file): 
        
        currently_written = 0
        
        count = 0   
        with open(str(input_file), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            count = sum(1 for row in f)
            
        with open(str(input_file), 'r') as infile, open(str(output_file), 'w') as outfile:
            wikidata_base = 'http://www.wikidata.org/entity/'
            wikipedia_base = 'https://en.wikipedia.org/wiki/'
            base_len = len(wikidata_base)
            reader = csv.reader(infile, delimiter=',')
            writer = csv.writer(outfile, delimiter=',')
            next(reader, None) 
            i = 0
            while i < count:
                row = next(reader)
                data_url = row[0]
                data_id = data_url[base_len:]
                title = row[1]
                title = title.replace(' ', '_')
                wikipedia_url = wikipedia_base + title
                try:
                    html = urllib.request.urlopen(wikipedia_url).read()
                    writer.writerow([data_id, title, wikipedia_url, data_url])
                    self.total_written += 1
                    currently_written += 1
                    percent = round(i/count, 4)
                    sys.stdout.write('\r'+ 'percent_complete: ' + str(percent) + ' currently_written: ' + str(currently_written) + ' total written: ' \
                                     + str(self.total_written))
                except:
                    self.total_skipped +=1
                    percent = round(i/count, 4)
                    sys.stdout.write('\r'+ 'percent_complete: ' + str(percent) + ' currently_written: ' + str(currently_written) + ' total written: ' \
                                     + str(self.total_written))
                i += 1