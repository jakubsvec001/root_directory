import os
import time
import sys
import csv
import pandas as pd
from selenium.webdriver import Firefox
from timeit import default_timer
from urllib.parse import unquote


import os
import time
import sys
import csv
import pandas as pd
from selenium.webdriver import Firefox
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
                if button.is_displayed():
                    button.click()
            time.sleep(3)
            expand_buttons = self._get_expand_buttons()
            end = default_timer()
            print(f'depth of { self.depth } took {str(round((end-start)/60, 2))} minutes to open')
            self.depth += 1
        if self.save=='csv':
            start = default_timer()
            self.save_csv()
            end = default_timer()
            print(str(round((end-start)/60, 2)) + f' minutes to save to csv')
            
    def save_csv(self):
        if self.all_pages==True:
            with open(f'seed_pages/all_pages_{ self.category }_d{ self.depth }.csv', 'w') as out:
                writer = csv.writer(out, lineterminator='\n')
                for a in self.browser.find_elements_by_xpath('.//a'):
                    writer.writerow([a.get_attribute('href')])
        else:
            with open(f'seed_categories/cat_pages_{ self.category }_d { self.depth }.csv', 'w') as out:
                writer = csv.writer(out, lineterminator='\n')
                for a in self.browser.find_elements_by_xpath('.//a'):
                    writer.writerow([a.get_attribute('href')])    
    
    def scrape(self, category, search_depth=3, save='csv'):
        """Scrape for either categories or all categories and pages"""
        self.category = category.replace(' ','_')
        self.search_depth = search_depth
        self.save = save
        self.browser = Firefox()
        print(self.all_pages)
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
        
# def main():
#     """Open the page and expand all categories."""
#     browser = Firefox()
#     browser.get('https://en.wikipedia.org/wiki/Special:CategoryTree?target=mathematics&mode=all&namespaces=&title=Special%3ACategoryTree')
#     expand_all_categories(browser, 'physics', search_depth=3, save='csv')