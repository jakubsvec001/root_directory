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


class WikiFinder(object):
    """Parse Wikipedia data dump files located at 
    https://dumps.wikimedia.org/enwiki/latest/
    one at a time searching for page tags"""

    def __init__(self, titles_csv, target=None, save=True, page_limit=None):
        self.titles_to_find = pd.read_csv(titles_csv, sep='\t', encoding='utf-8')['cleaned_url'].values
        self.target = target
        self.save = save
        self.page_limit = page_limit

    def create_corpus(self, filein, target):
        '''Return a list of articles in a dictionary format OR
        save articles to a mongodb database'''
        self.target = target
        start = timer()
        lines = self._get_lines_bz2(filein)
        pages = self._find_pages(lines)
        if self.save:
            mc = MongoClient()
            db = mc['wiki_cache']
            collection = db['pages']
            for page in pages:
                cached_article = collection.find_one({'title': page['title']})
                if cached_article is None:
                    collection.insert_one(page)
        else:
            pages = list(pages)
            return pages
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
            if self.page_limit and i >= self.page_limit:
                break

    def _find_pages(self, lines):
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
                parsed = self._identify_page(raw_xml)
                if parsed:
                    found_count += 1
                    yield parsed
                page = []
                sys.stdout.write('\r' + f'Found articles: {found_count} Search count: {search_count}')
                if self.page_limit:
                    if found_count >= self.page_limit:
                        break
            elif inpage:
                page.append(line)

    def _identify_page(self, raw_xml):
        """Indentify whether or not article is in self.titles_to_find"""        
        soup = bs(raw_xml, 'lxml')
        title = soup.select_one('title').text
        if title in self.titles_to_find:
            return {'title': title,
                    'full_raw_xml': raw_xml,
                    'target': self.target,
                    }



class CatFinder(object):
    """Parse Wikipedia data dump files located at 
    https://dumps.wikimedia.org/enwiki/latest/
    one at a time searching for category edges"""

    def __init__(self, save=True, page_limit=None):
        self.save = save
        self.page_limit = page_limit

    def create_edgelist(self, filein):
        start = timer()
        lines = self._get_lines_bz2(filein)
        pages = self._find_all_edges(lines)
        if self.save:
            mc = MongoClient()
            db = mc['wiki_cache']
            collection = db['edgelist']
            for title, categories in pages:
                if categories != []:
                    cached_child = collection.find_one({'parent_categories': [title, categories]})
                    if cached_child is None:
                        collection.insert_one({'parent_categories': [title, categories]})
        else:
            pages = list(pages)
            return pages
        end = timer()
        time = round((end - start) / 60)
        stopwatch = f'It took {time} minutes to complete the search'
        return stopwatch

    def _find_all_edges(self, lines):
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
                title, categories = self._parse_edges(raw_xml)
                if categories:
                    found_count += 1
                    yield title, categories
                page = []
                sys.stdout.write('\r' + f'Found articles: {found_count} Search count: {search_count}')
                if self.page_limit:
                    if found_count >= self.page_limit:
                        break
            elif inpage:
                page.append(line)


    def _get_lines_bz2(self, filename): 
        """yield each uncompressed line from bz2 file"""
        for i, line in enumerate(subprocess.Popen(['bzcat'], 
                                    stdin = open(filename, 'rb'), 
                                    stdout = subprocess.PIPE
                                    ).stdout):
            yield line.decode()
            if self.page_limit and i >= self.page_limit:
                break


    def _parse_edges(self, raw_xml):
        # Find category markup:
        re_categories = re.compile(r'\[\[([cC]ategory:[^][]*)\]\]', re.UNICODE) 
        categories = re_categories.findall(raw_xml)
        soup = bs(raw_xml, 'lxml')
        title = soup.select_one('title').text
        return title, categories
        