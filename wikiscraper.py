import os
import time
import sys
import csv
import requests
import pandas as pd
from timeit import default_timer
from urllib.parse import unquote
from bs4 import BeautifulSoup as bs
from selenium.webdriver import Firefox

class TreeScraper(object):
    """Scrape Wikipedia's Special Category Tree page: https://en.wikipedia.org/wiki/Special:CategoryTree
    
    EXAMPLE USE:
    import wikiscraper
    scraper = wikiscraper.WikiTreeScraper()
    scraper.scrape(category='mathematics', search_depth=3, save='csv')
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
        self.atag = soup.find_all('a', class_='CategoryTreeLabel')
        self.href_count = len(self.atag)
        if self.save=='csv':
            start = default_timer()
            self._save_csv()
            end = default_timer()
            print(str(round((end-start)/60, 2)) + f' minutes to save to csv')
        elif self.save==False:
            return self._href_list()
            

    def _save_csv(self):
        """Save to a csv file"""
        # Save pages and categories to a 'seed_pages' dir
        if self.all_pages==True:
            with open(f'seed_pages/all_pages_{ self.category }_d{ self.depth }_{ self.href_count }.csv', 'w') as out:
                writer = csv.writer(out, lineterminator='\n')
                for a in self.atag:
                    writer.writerow([a['href']])
        # Save only categories to a 'seed_categories' dir
        else:
            with open(f'seed_categories/cat_pages_{ self.category }_d { self.depth }_{ self.href_count }.csv', 'w') as out:
                writer = csv.writer(out, lineterminator='\n')
                for a in self.atag:
                    writer.writerow([a['href']])
    
    def _href_list(self):
        """save hrefs as an attribute"""
        self.hrefs = []
        for a in self.atag:
            self.hrefs.append(a['href'])
    
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


class VitalScraper(object):
    """Scrape the highly curated "Wikipedia:Vital articles" pages
    https://en.wikipedia.org/wiki/Wikipedia:Vital_articles"""

    # Vital Articles Category Links:
    def __init__(self):    
        self.links_dict = dict(engineering = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5/Technology',
                    mathematics = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5/Mathematics',
                    physics = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5/Physical_sciences/Physics',
                    earth_science = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5/Physical_sciences/Earth_science',
                    chemistry = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5/Physical_sciences/Chemistry',
                    astronomy = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5/Physical_sciences/Astronomy',
                    arts = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5/Arts',
                    )

    def scrape(self, category):
        """scrape all '/wiki/' links from the given category and its associated url"""
        url = self.links_dict[category]
        response = requests.get(url)
        soup = bs(response.content, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            # links.append(a['href'])
            link = self.filter_links(a['href'])
            if link:
                links.append(link)
        return links
    
    def filter_links(self, link):
        """consume a link and return if link does not contain excluded items"""
        exclude_list = ['Wikipedia:Vital_articles', 
                        'Template:', 
                        'Special:', 
                        'Featured_articles',
                        'Good_articles',
                        'General_disclaimer',
                        'User:',
                        'Portal',
                        'Help:',
                        'Wikipedia:Community_portal',
                        'Category:',
                        '/Main_Page',
                        'Wikipedia_talk:',
                        'Wikipedia:',
                        'Template_talk:']
        flag = False
        for item in exclude_list:
            if item in link:
                flag = True
        if link.startswith(('#', '//', 'https:')):
            flag = True
        if flag == False:
            return link