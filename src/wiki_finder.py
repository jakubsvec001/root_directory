import pandas as pd
import subprocess
import mwparserfromhell
import glob
import os
import sys
import re
import json
from timeit import default_timer as timer
from bs4 import BeautifulSoup as bs
from pymongo import MongoClient
from gensim.corpora import wikicorpus


class WikiFinder(object):
    """Parse Wikipedia data dump files located at
    https://dumps.wikimedia.org/enwiki/latest/
    one at a time searching for page tags
        ----------
        Parameters
        ----------

        Returns
        -------

    """

    def __init__(self, titles_csv, target=None, save=True, page_limit=None):
        self.titles_to_find = pd.read_csv(
            titles_csv, sep='\t', encoding='utf-8')['cleaned_url'].values
        self.target = target
        self.save = save
        self.page_limit = page_limit

    def create_corpus(self, filein, target):
        """Return a list of articles in a dictionary format OR
        save articles to a mongodb database"""
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
                                 stdin=open(filename, 'rb'),
                                 stdout=subprocess.PIPE).stdout):
            yield line.decode()
            if self.page_limit and i >= self.page_limit:
                break

    def _find_pages(self, lines):
        """yield each page parsed from a wikidump"""
        found_count = 0
        search_count = 0
        for raw_xml in self._find_pages_raw_xml(lines):
            parsed = self._identify_page(raw_xml)
            if parsed:
                found_count += 1
                yield parsed
                sys.stdout.write('\r' + f'Found articles: {found_count} ' +
                                 'Search count: {search_count}')
                if self.page_limit:
                    if found_count >= self.page_limit:
                        break

    def _find_pages_raw_xml(self, lines):
        """Yield raw xml for each page from a wikidump."""
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
                yield raw_xml
                page = []
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
    one at a time searching for category edges
        ----------
        Parameters
        ----------

        Returns
        -------

    """

    def __init__(self, save=True, page_limit=None):
        self.save = save
        self.page_limit = page_limit

    def create_edgelist(self, filein, output_collection):
        start = timer()
        lines = self._get_lines_bz2(filein)
        pages = self._find_all_edges(lines)
        if self.save:
            mc = MongoClient()
            db = mc['wiki_cache']
            collection = db[output_collection]
            for title, categories in pages:
                if categories != []:
                    collection.insert_one({'parent_categories':
                                          [title, categories]})
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
                sys.stdout.write('\r' + f'Found articles: {found_count} ' +
                                 'Search count: {search_count}')
                if self.page_limit:
                    if found_count >= self.page_limit:
                        break
            elif inpage:
                page.append(line)

    def _get_lines_bz2(self, filename):
        """yield each uncompressed line from bz2 file"""
        for i, line in enumerate(subprocess.Popen(['bzcat'],
                                 stdin=open(filename, 'rb'),
                                 stdout=subprocess.PIPE
                                 ).stdout):
            yield line.decode()
            if self.page_limit and i >= self.page_limit:
                break

    def _parse_edges(self, raw_xml):
        # Find category markup:
        re_categories = re.compile(r'\[\[([cC]ategory:[^][]*)\]\]',
                                   re.UNICODE)
        categories = re_categories.findall(raw_xml)
        soup = bs(raw_xml, 'lxml')
        title = soup.select_one('title').text
        return title, categories


def get_lines_bz2(filename, limit=None):
    """yield each uncompressed line from bz2 file
        ----------
        Parameters
        ----------

        Returns
        -------

    """
    for i, line in enumerate(subprocess.Popen(['bzcat'],
                             stdin=open(filename, 'rb'),
                             stdout=subprocess.PIPE
                             ).stdout):
        yield line.decode()
        if limit and i >= limit:
            break


def page_generator_articles_only(lines, limit=None):
    """yield each page from wiki_dump
        ----------
        Parameters
        ----------

        Returns
        -------

    """
    search_count = 0
    page = []
    inpage = False
    for line in lines:
        line = line.lstrip()
        if line.startswith('<page>'):
            inpage = True
        elif line.startswith('</page>'):
            inpage = False
            raw_xml = ''.join(page)
            soup = bs(raw_xml, 'lxml')
            namespaces = ['0']
            if soup.select_one('ns').text in namespaces:
                if '#REDIRECT' not in soup.select_one('text').text:
                    search_count += 1
                    yield raw_xml
            page = []
            # sys.stdout.write('\r' + f'Search count: {search_count}')
            if limit:
                if search_count >= limit:
                    break
        elif inpage:
            page.append(line)


def identify_page(raw_xml):
    """Indentify whether or not article is in self.titles_to_find
        ----------
        Parameters
        ----------

        Returns
        -------

    """
    soup = bs(raw_xml, 'lxml')
    title = soup.select_one('title').text
    return {'title': title,
            'full_raw_xml': raw_xml,
            }