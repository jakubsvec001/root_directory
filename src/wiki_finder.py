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
        # Find math content:
        re_math = re.compile(r'<math([> ].*?)(</math>|/>)', re.DOTALL|re.UNICODE)
        # Find all other tags:
        re_all_tags = re.compile(r'<(.*?)>', re.DOTALL|re.UNICODE)
        # Find category markup:
        re_categories = re.compile(r'\[\[Category:[^][]*\]\]', re.UNICODE)
        # rm File and Image templates:
        re_rm_file_image = re.compile(r'\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)
        # Capture interlinks text and article linked:
        re_interlinkstext_link = re.compile(r'\[{2}(.*?)\]{2}', re.UNICODE)
        # Simplify links, keep description:
        re_simplify_link = re.compile(r'\[([^][]*)\|([^][]*)\]', re.DOTALL|re.UNICODE)
        # Keep image Description:
        re_image_description = re.compile(r'\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
        # Keep file descirption:
        re_file_description = re.compile(r'\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
        # External links:
        re_external_links = re.compile(r'<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL|re.UNICODE)
        
        soup = bs(raw_xml, 'lxml')
        title = soup.select_one('title').text
        if title in self.titles_to_find:
            return {'title': title,
                    'full_raw_xml': raw_xml,
                    'target': self.target,
                    }

        
            # id_ = soup.select_one('id').text
            # markup_text = soup.select_one('text').text
            # #use regex to delete 'Category' tags and text from raw_xml
            # cleaned_text = []
            # kw = ('[[Category:', 'thumb')
            # for line in markup_text.split('\n'):
            #     if line.startswith(kw):
            #         continue
            #     cleaned_text.append(line)
            # categories = re_categories.findall(raw_xml)
            # tags = re_all_tags.findall(raw_xml)
            # file_desc = re_file_description.findall(raw_xml)
            # if file_desc != []:
            #     file_desc = ' '.join(file_desc[0][1:])[1:]
            #     file_desc = wikicorpus.remove_markup(file_desc)
            # image_desc = re_image_description.findall(raw_xml)
            # if image_desc != []:
            #     image_desc = ' '.join(image_desc[0][2:])
            #     image_desc = wikicorpus.remove_markup(image_desc)
            # external_links = re_external_links.findall(raw_xml)
            # simple_links = re_simplify_link.findall(raw_xml)
            # interlinks = re_interlinkstext_link.findall(raw_xml)
            # math = wikicorpus.RE_P10.findall(markup_text)

            # for category in categories:
            #     raw_xml = raw_xml.replace(category, ' ')
            # timestamp = soup.select_one('timestamp').text
            # wiki = mwparserfromhell.parse(markup_text)

            # wikilinks = []
            # for link in wiki.filter_wikilinks():
            #     link = link.strip('[]')
            #     if 'File:' not in link and \
            #     'Category:' not in link and \
            #         'Wikipedia:' not in link and \
            #         'en:' not in link and \
            #         'Image:' not in link:
            #         wikilinks.append(link)

            # return {
            #     'title': title,
            #     'full_raw_xml': raw_xml,
            #     'target': self.target,
            #     }



def multi_process_corpus(self, dump_file, title_file, dump_search_limit=None):
    """creates a multiprocessing pool to search multiple
    files with multiple workers."""
    start = timer()
    input_titles = title_file
    dump_list = glob.glob(dump_file + '*.bz2')
    input_titles = pd.read_csv(title_file, sep='\t', encoding='utf-8')['cleaned_url'].tolist()
    pool = Pool(processes = os.cpu_count())

    # Map (service, tasks), applies function to each partition
    if not dump_search_limit:
        pool.map(self.create_corpus, dump_list)
    else:
        pool.map(self.create_corpus, dump_list[:dump_search_limit])

    pool.close()
    pool.join()

    end = timer()
    stopwatch = round((start - end)/60, 2) 
    print(f'{stopwatch} seconds elapsed.')   


# if __name__ == '__main__':
    
#     start = timer()
#     n = 8
#     dumps = glob.glob(str(sys.argv[1]) + '*.bz2')
#     input_titles = sys.argv[2]
#     target = sys.argv[3]

#     input_titles = pd.read_csv(str(input_titles), sep='\t', encoding='utf-8', header=None)[0].tolist()
#     if len(sys.argv) != 4:
#         print(f"Usage: python wiki_parse.py <dump directory> <target article file> <topic target name>\nArgs given: {len(sys.argv)}")
#         sys.exit(1)
    
#     pool = Pool(processes = os.cpu_count()) 
#     results = pool.map(create_corpus, dumps[:n])

#     pool.close()
#     pool.join()

#     end = timer()
#     stopwatch = round(((start - end) / 60), 2)
#     print(f'{stopwatch} seconds elapsed.')