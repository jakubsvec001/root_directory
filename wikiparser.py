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


class DumpParser(object):
    """Parse Wikipedia data dump files located at 
    https://dumps.wikimedia.org/enwiki/latest/
    one at a time searching for page tags"""

    def __init__(self, input_titles, target, save=True, limit=None, dir_out='temp_results'):
        """initialize with list of input titles to match articles on, 
        plus add the target classification of the articles"""
        self.target = target
        self.save = save
        self.input_titles = input_titles
        self.limit = limit
        self.dir_out = dir_out

                
    def create_corpus(self, filein):
        '''Return a list of articles in a dictionary format OR
        save articles to a mongodb database'''
        start = timer()
        name_str = filein.partition('-')[-1].split('.')[-3]
        lines = self._get_lines_bz2(filein)
        pages = self._find_pages(lines)
        # CONTIUNE HERE TO SAVE INTO MONGODB
        # if save:
        #     mc = MongoClient()
        #     db = mc['cache']
        #     math = db['math']
        #     for page in pages:
        #         cached_article = math.find_one({'title': page['title']})
        #         if cashed_article is None:
        #             math.insert_one(page)
        # else:
        #     pages = list(pages)
        #     return pages
        end = timer()
        time = round((end - start) / 60)
        stopwatch = f'It took {time} minutes to complete the search'
        print(' SAVED TO: ' + self.dir_out + name_str + '.json')
        return stopwatch

    def multi_process_corpus(self, dump_file, title_file):
        """creates a multiprocessing pool to search multiple
        files with multiple workers."""
        start = timer()
        global input_titles
        input_titles = title_file
        dump_list = glob.glob(dump_file + '*.bz2')
        input_titles = pd.read_csv(title_file, sep='\t', encoding='utf-8', header=None)[0].tolist()
        pool = Pool(processes = os.cpu_count())

        # Map (service, tasks), applies function to each partition
        pool.map(self.create_corpus, dump_list[:4])

        pool.close()
        pool.join()

        end = timer()
        stopwatch = round((start - end)/60, 2) 
        print(f'{stopwatch} seconds elapsed.')

    def _get_lines_bz2(self, filename): 
        """yield each uncompressed line from bz2 file"""
        for i, line in enumerate(subprocess.Popen(['bzcat'], 
                                    stdin = open(filename, 'rb'), 
                                    stdout = subprocess.PIPE
                                    ).stdout):
            yield line.decode()
            if self.limit and i >= self.limit:
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
                parsed = self._parse_page(raw_xml)
                if parsed:
                    found_count += 1
                    yield parsed
                page = []
                sys.stdout.write('\r' + f'Found articles: {found_count} Search count: {search_count}')
                if self.limit:
                    if found_count >= self.limit:
                        break
            elif inpage:
                page.append(line)

    def _parse_page(self, raw_xml):
        """Return a dict of page content 
        title
        timestamp
        id
        raw_xml
        markup_text
        cleaned_text
        links
        target
        """
        
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
        if title in input_titles:
            id = soup.select_one('id').text
            markup_text = soup.select_one('text').text
            #use regex to delete 'Category' tags and text from raw_xml
            cleaned_text = []
            kw = ('[[Category:', 'thumb')
            for line in markup_text.split('\n'):
                if line.startswith(kw):
                    continue
                cleaned_text.append(line)
            categories = re_categories.findall(raw_xml)
            tags = re_all_tags.findall(raw_xml)
            file_desc = re_file_description.findall(raw_xml)
            if file_desc != []:
                file_desc = ' '.join(file_desc[0][1:])[1:]
                file_desc = wikicorpus.remove_markup(file_desc)
            image_desc = re_image_description.findall(raw_xml)
            if image_desc != []:
                image_desc = ' '.join(image_desc[0][2:])
                image_desc = wikicorpus.remove_markup(image_desc)
            external_links = re_external_links.findall(raw_xml)
            simple_links = re_simplify_link.findall(raw_xml)
            interlinks = re_interlinkstext_link.findall(raw_xml)
            math = wikicorpus.RE_P10.findall(markup_text)

            for category in categories:
                raw_xml = raw_xml.replace(category, ' ')
            timestamp = soup.select_one('timestamp').text
            wiki = mwparserfromhell.parse(markup_text)
            

            wikilinks = []
            for link in wiki.filter_wikilinks():
                link = link.strip('[]')
                if 'File:' not in link and \
                'Category:' not in link and \
                    'Wikipedia:' not in link and \
                    'en:' not in link and \
                    'Image:' not in link:
                    wikilinks.append(link)

            return {
                'title': title,
                'timestamp': timestamp ,
                'id': id, 
                'full_raw_xml': raw_xml,
                'full_markup_text': ''.join(markup_text),
                'cleaned_markup_text': ' '.join(cleaned_text),
                'links': wikilinks,
                'target': self.target,
                'categories': categories,
                'tags': tags,
                'file_desc': file_desc,
                'image_desc': image_desc,
                'external_links': external_links,
                'simple_links': simple_links,
                'interlinks': interlinks,
                'math': math,}

    

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