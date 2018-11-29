from bs4 import BeautifulSoup as bs
from bson.objectid import ObjectId
import mwparserfromhell
import re
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import glob
from pymongo import MongoClient
from gensim.corpora import wikicorpus
import sys


def disect_update_database(db_name, collection_name):
    """parse and update a mongodb database with cleaned xml"""
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    document_generator = mongodb_page_stream(collection)
    count = 0
    for document in document_generator:
        features = disect_page(document['title'], document['full_raw_xml'])
        _update_document(collection, features)
        count +=1
        sys.stdout.write('\r'+ f'YAAAAAAAS! Updated {count} documents: Title = {document["title"]}')


def mongodb_page_stream(collection):
    """yield a new page from a mongodb collection"""
    document_generator = collection.find()
    return document_generator


def disect_page(title, xml):
    """parse raw wikipedia xml"""
    # extract links from xml
    links = get_links(xml)
    # extract image text from xml
    image_desc = clean_image_desc(xml)
    # extract file text from xml
    file_desc = clean_file_desc(xml)
    # extract the categories
    categories = re_categories.findall(xml)
    # clean xml of category information
    clean_xml = replace_categories_in_xml(xml, categories)
    # clean categories of Category: links
    cleaned_links = clean_links_list(links)
    # make BeautifulSoup object from cleaned xml
    soup = bs(clean_xml, 'lxml')
    # # extract the title out
    # page_title = soup.select_one('title').text
    # extract the timestamp
    timestamp = soup.select_one('timestamp').text
    # extract headers
    headers = get_headers(soup.text)
    # convert to markup, remove markup text, strip remaining text, replace newline characters
    markup_text = soup.select_one('text').text
    text_remove_markup = wikicorpus.remove_markup(markup_text)
    text_strip_code = mwparserfromhell.parse(text_remove_markup).strip_code()
    clean_text = replace_multiple(text_strip_code, ['\n', '(', ')', ',', ';', '[', ']', '"', ':'], ' ')
    feature_union = join_features(title, headers, clean_text, cleaned_links)
    return {'title': title, 
            'clean_text': clean_text,
            'timestamp': timestamp, 
            'headers': headers, 
            'clean_links':cleaned_links,
            'parent_categories': {title: categories},
            'feature_union': feature_union}


def replace_multiple(main_string, to_be_replaced, new_string):
    """replace extra elements in a text string"""
    for elem in to_be_replaced :
        if elem in main_string :
            main_string = main_string.replace(elem, new_string)
    return  main_string


def _update_document(collection, features):
    """Used by disect_update_database(). Updates a document in mongodb"""
    collection.update_one({'title': features['title']}, {'$set': 
                                                           {'feature_union': features['feature_union'],
                                                            'parent_categories': features['parent_categories']}})


def num_target_col_db(db_name, collection_name, target_name):
    """add a column to each document with a number and the target value
       for indexing train/test split. Uses _check_target_presence_in_db()"""
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    _check_target_presence_in_db(collection, target_name)
    docs = collection.find()
    for idx, doc in enumerate(docs):
        _id = doc['_id']
        if doc['target'] == target_name:
            target = 1
        else:
            target = 0
        collection.update({'_id': ObjectId(_id)}, {'$set' : 
                                                            {'idx/target': (idx, target)}})


def _check_target_presence_in_db(collection, target_name):
    """check for presence of target in db 
    before adding num_target column"""
    docs = collection.find()
    hit = False
    for doc in docs:
        if doc['target'] == target_name:
            hit = True
            break
    if hit == False:
        raise ValueError('OH NOOO! No targets saved. '+
                         'Check your target_name parameter')


def join_features(page_title, headers, clean_text, cleaned_links):
    return ' '.join([page_title] + [clean_text] + cleaned_links + headers)


def get_links(xml):
    links = re_interlinkstext_link.findall(xml)
    clean_links = []
    for link in links:
        if '|' in link:
            clean_links.append(link.partition('|')[0].replace('#', ' '))
        else: 
            clean_links.append(link.replace('#',' '))
    return clean_links


def clean_image_desc(xml):
    image_desc = re_image_description.findall(xml)
    if image_desc != []:
        image_desc = ' '.join(image_desc[0][2:])
        image_desc = wikicorpus.remove_markup(image_desc)
    return image_desc


def clean_file_desc(xml):
    file_desc = re_file_description.findall(xml)
    if file_desc != []:
        file_desc = ' '.join(file_desc[0][1:])[1:]
        file_desc = wikicorpus.remove_markup(file_desc)
    return file_desc


def replace_categories_in_xml(xml, categories):
    if categories == [] or xml == []:
        return xml
    for category in categories:
        cleaned_xml = xml.replace(category, ' ')
    return cleaned_xml


def clean_links_list(links_list):
    clean_links = []
    for link in links_list:
        link = link.replace('(disambiguation)', '')
        if 'File:' not in link and \
           'Category:' not in link and \
           'Wikipedia:' not in link and \
           'en:' not in link and \
           'User:' not in link and \
           'Template:' not in link and \
           'User talk:' not in link and \
           'Special:' not in link and \
           'Project:' not in link and \
           'WP:' not in link and \
           'd:' not in link and \
           'Image:' not in link:
            clean_links.append(link.replace('&amp;', ' ').strip())
    scrubbed_links = []
    for link in clean_links:
        scrubbed_links.append(replace_multiple(link, 
                                ['\n', '(', ')', ',', ';', '[', ']', '"', ':'], 
                                ' '))
    return scrubbed_links


def get_headers(text):
    headers = []
    lines = text.split('\n')
    for line in lines:
        if line.startswith('='):
            header = line.replace('=', '').strip()
            headers.append(mwparserfromhell.parse(header).strip_code())
    clean_headers = []
    for header in headers:
        clean_headers.append(replace_multiple(header, 
                                ['\n', '(', ')', ',', ';', '[', ']', '"', ':'], 
                                ' '))
    return clean_headers


# Find math content:
re_math = re.compile(r'<math([> ].*?)(</math>|/>)', re.DOTALL|re.UNICODE)
# Find all other tags:
re_all_tags = re.compile(r'<(.*?)>', re.DOTALL|re.UNICODE)
# Find category markup:
re_categories = re.compile(r'\[\[([cC]ategory:[^][]*)\]\]', re.UNICODE) 
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