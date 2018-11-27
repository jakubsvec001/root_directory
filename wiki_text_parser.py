from bs4 import BeautifulSoup as bs
from bson.objectid import ObjectId
from urllib.parse import unquote
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
import numpy as np
import glob
from pymongo import MongoClient
from gensim.corpora import wikicorpus


def mongodb_page_stream(db_name, collection_name):
    """yield a new page from a mongodb collection"""
    mc = MongoClient()
    db = mc[db_name]
    collection = db[collection_name]
    document_generator = collection.find()
    return document_generator

def parse_update_database(db_name, collection_name):
    """parse and update a mongodb database with cleaned xml"""
    document_generator = mongodb_page_stream(db_name, collection_name)
    for document in document_generator:
        yield parse_page(document['full_raw_xml'])

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

def parse_page(xml):
    """parse raw wikipedia xml"""
    # grab the raw xml from the document
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
    # extract the title out
    page_title = soup.select_one('title').text
    # extract the timestamp
    timestamp = soup.select_one('timestamp').text
    # extract headers
    headers = get_headers(soup.text)
    # convert to markup, remove markup text, strip remaining text, replace newline characters
    markup_text = soup.select_one('text').text
    text_remove_markup = wikicorpus.remove_markup(markup_text)
    text_strip_code = mwparserfromhell.parse(text_remove_markup).strip_code()
    clean_text = text_strip_code.replace('\n','')
    return {'clean_text': clean_text,'timestamp': timestamp, 'headers': headers, 'clean_links':cleaned_links, 'parent_categories': {page_title: categories}}

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
    return clean_links

def get_headers(text):
    headers = []
    lines = text.split('\n')
    for line in lines:
        if line.startswith('='):
            header = line.replace('=', '').strip()
            headers.append(mwparserfromhell.parse(header).strip_code())
    return headers

def update_document():
    pass