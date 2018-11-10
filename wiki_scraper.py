import os
import time
import sys
import csv
import pandas as pd
from selenium.webdriver import Firefox
from bs4 import BeautifulSoup as bs
from timeit import default_timer
from urllib.parse import unquote

class wikiScraper(object):
    """Scrape Wikipedia's Special Category Tree page: 
    https://en.wikipedia.org/wiki/Special:CategoryTree
    
    u

    EXAMPLE USE:
    scraper = wikiScrape()
    """
    
    def __init__(self, all_pages=True):
        self.all_pages = all_pages
        
        
    def _get_expand_buttons(self):
        """Return a list of expand buttons to click on."""
        return self.browser.find_elements_by_xpath("//span[@title='expand']")
    
    def _expand_all_categories(self):
        """Expand all categories on page."""
        start_time = default_timer()
        expand_buttons = self._get_expand_buttons()
        time.sleep(3)
        self.depth = 0
        print('num expand buttons ', len(expand_buttons))
        while self.depth < self.search_depth:
            start = default_timer()
            for button in expand_buttons:
                time.sleep(.05)
                if button.is_displayed():
                    button.click()
            time.sleep(3)
            expand_buttons = self._get_expand_buttons()
            end = default_timer()
            print(f'depth of { self.depth } took {str(round((end-start)/60, 2))} minutes to open')
            self.depth += 1
        html = self.browser.page_source
        soup = bs(html, 'html.parser')
        self.html = soup.find_all('a', class_='CategoryTreeLabel')
        self.href_count = len(self.html)
        if self.save=='csv':
            start = default_timer()
            self.save_csv()
            end = default_timer()
            print(str(round((end-start)/60, 2)) + f' minutes to save to csv')
            
    #  browser.page_source <--- converts to html that can be made into soup

    def save_csv(self):
        if self.all_pages==True:
            with open(f'seed_pages/all_pages_{ self.category }_d{ self.depth }_{ self.href_count }.csv', 'w') as out:
                writer = csv.writer(out, lineterminator='\n')
                for a in self.html:
                    writer.writerow([a['href']])
        else:
            with open(f'seed_categories/cat_pages_{ self.category }_d { self.depth }_{ self.href_count }.csv', 'w') as out:
                writer = csv.writer(out, lineterminator='\n')
                for a in self.html:
                    writer.writerow([a['href']])

    
    def scrape(self, category, search_depth=3, save='csv'):
        """Scrape for either categories or all categories and pages"""
        self.category = category.replace(' ','_')
        self.search_depth = search_depth
        self.save = save
        self.browser = Firefox()
        if self.all_pages==True:
            time.sleep(1)
            self.browser.get(f'https://en.wikipedia.org/wiki/Special:CategoryTree?target={ self.category }&mode=all&namespaces=&title=Special%3ACategoryTree')
            time.sleep(1)
            self._expand_all_categories()
        else:
            time.sleep(1)
            self.browser.get(f'https://en.wikipedia.org/wiki/Special:CategoryTree?target={ self.category }&mode=categories&namespaces=&title=Special%3ACategoryTree')
            time.sleep(1)
            self._expand_all_categories()
        