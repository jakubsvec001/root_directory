from bs4 import BeautifulSoup as bs
import subprocess
import os
import sys
import mwparserfromhell
import re
import json
from timeit import default_timer as timer
from multiprocessing import Pool 
import tqdm 
from itertools import chain
from functools import partial
import pandas as pd
import glob
from pymongo import MongoClient
from gensim.corpora import wikicorpus


class CategoryFinder(object):
    """Parse Wikipedia data dump files located at 
    https://dumps.wikimedia.org/enwiki/latest/
    one at a time searching for page tags"""

    def __init__(self, titles_to_find=None, save=True):
        self.titles_to_find = titles_to_find
        self.save = save
        self.scan_limit = None

    def create_links_csv(self, filein, scan_limit=None):
        '''Return a list of articles in a dictionary format OR
        save articles to a mongodb database'''
        self.scan_limit = scan_limit
        name_str = filein.partition('-')[-1].split('.')[-3]
        start = timer()
        lines = self._get_lines_bz2(filein)
        pages = self._examine_pages(lines)
        if self.save:
            data = pd.DataFrame(pages)
            data.to_csv(f'category_data/{name_str}_links.csv', index=False, )
        end = timer()
        time = round((end - start) / 60)
        stopwatch = f'It took {time} minutes to complete the search'
        return stopwatch

    def _get_lines_bz2(self, filename): 
        """yield each uncompressed line from bz2 file"""
        for i, line in enumerate(subprocess.Popen(['bzcat'], 
                                    stdin = open(filename, 'rb'), 
                                    stdout = subprocess.PIPE
                                    ).stdout):
            yield line.decode()
            if self.scan_limit and i >= self.scan_limit:
                break

    def _examine_pages(self, lines):
        """yield each page from a wikidump"""
        found_count = 0
        search_count = 0
        page = []
        inpage = False
        for line in lines:
            line = line.lstrip()
            if line.startswith('<page>'):
                inpage = True
            elif line.startswith('</page>'):
                search_count += 1
                inpage = False
                raw_xml = ''.join(page)
                categories = self._get_category_info(raw_xml)
                if categories:
                    found_count += 1
                    yield categories
                page = []
                sys.stdout.write('\r' + f'Found category links: {found_count} Search count: {search_count}')
            elif inpage:
                page.append(line)

    def _get_category_info(self, raw_xml):
        """find category links for page"""
        # Find category markup:
        re_categories = re.compile(r'\[\[(Category:[^][]*)\]\]', re.UNICODE)
        categories = re_categories.findall(raw_xml)
        cleaned_cats = []
        soup = bs(raw_xml, 'lxml')
        title = soup.select_one('title').text
        if categories:
            return title, categories