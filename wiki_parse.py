from bs4 import BeautifulSoup as bs
import subprocess
import os
import sys
import mwparserfromhell
import re
import json
from timeit import default_timer as timer


# re_math = re.compile(r'<math([> ].*?)(</math>|/>)', re.DOTALL|re.UNICODE)
# re_all_other_tags = re.compile(r'<(.*?)>', re.DOTALL|re.UNICODE)
# re_categories = re.compile(r'\[\[Category:[^][]*\]\]', re.UNICODE)
# re_rm_file_image_temp = re.compile(r'\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)
# re_rm_url_keep_description = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)
# re_simplify_links = re.compile(r'\[([^][]*)\|([^][]*)\]', re.DOTALL|re.UNICODE)
# re_keep_image_description = re.compile(r'\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
# re_infobox = re.compile(r'(?=\{Infobox)(\{([^{}]|(?1))*\})', re.UNICODE)
# re_remove_url = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)

def get_lines_bz2(filename): 
    """yield each uncompressed line from bz2 file"""
    for line in subprocess.Popen(['bzcat'], 
                             stdin = open(filename, 'rb'), 
                             stdout = subprocess.PIPE
                             ).stdout:
        yield line.decode()

def find_pages(lines, input_titles, limit=None):
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
            text = ''.join(page)
            parsed = parse_page(text, input_titles)
            if parsed:
                found_count += 1
                yield parsed
            page = []
            sys.stdout.write('\r' + f'Found articles: {found_count} Search count: {search_count}')
            if limit:
                if found_count >= limit:
                    break
        elif inpage:
            page.append(line)


def parse_page(page, input_titles):
    """Return a dict of page content"""
    soup = bs(page, 'lxml')
    title = soup.select_one('title').text
    page_content = soup.select_one('text').text
    timestamp = soup.select_one('timestamp').text
    if title in input_titles:
        wiki = mwparserfromhell.parse(page_content)
        cleaned_text = strip_stripped_code(wiki) 
        tags = wiki.filter_tags()
        wikilinks = []
        for link in wiki.filter_wikilinks():
            link = link.strip('[]')
            if 'File:' not in link and \
               'Category:' not in link and \
                'Wikipedia:' not in link and \
                'en:' not in link and \
                'Image:' not in link:
                wikilinks.append(link)
        return {'title': title, 
                'stripped_content': cleaned_text, 
                'full_content': page_content,
                'wikilinks': wikilinks,
                # 'tags': tags,
                'timestamp': timestamp}


def create_corpus(file, input_titles, save=True, limit=None):
    '''Return a list of articles in a dictionary format'''
    start = timer()
    lines = get_lines_bz2(file)
    pages = find_pages(lines, input_titles, limit)
    if save:
        fout = open('data/temp_results/file1.json', 'w')
        save_json(fout, pages)
        fout.close()
    else:
        pages = list(pages)
    end = timer()
    time = round((end - start) / 60)
    stopwatch = f'It took {time} minutes to complete the search'
    return stopwatch, pages

def strip_stripped_code(wiki):
    stripped = wiki.strip_code()
    cleaned = []
    kw = ('Category:', 'thumb')
    for line in stripped.split('\n'):
        if line.startswith(kw):
            continue
        cleaned.append(line)
    return cleaned    
        
# if __name__ == '__main__':
    
#     if len(sys.argv) != 2:
#         print('Usage: python check_wiki_corpus.py <corpus_file>')
#         sys.exit(1)

#     corpus_file = open(sys.argv[1],'r')
#     check_corpus(corpus_file)
#     corpus = load_corpus(corpus_file)
    

# soup = bs(b'\n'.join(lines).decode('utf-8'), 'lxml')

def save_json(fout, pages):
    for page in pages:
        fout.write(json.dumps(page) + '\n')

def load_json(filename):
    data = []
    with open('data/temp_results/file1.json', 'r') as fin:
        for l in fin.readlines():
            data.append(json.loads(l))
    return data