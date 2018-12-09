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
    """Scrape Wikipedia's Special Category Tree page:
    https://en.wikipedia.org/wiki/Special:CategoryTree
        ----------
        Parameters
        ----------
        
        Returns
        -------
        
    """

    def __init__(self, all_pages=True):
        self.all_pages = all_pages

    def _get_expand_buttons(self):
        """Return a list of expand buttons to click on.
            ----------
            Parameters
            ----------

            Returns
            -------

        """
        return self.browser.find_elements_by_xpath("//span[@title='expand']")

    def _expand_all_categories(self):
        """Expand all categories on page.
            ----------
            Parameters
            ----------

            Returns
            -------

        """
        self.depth = 0
        self.df = pd.DataFrame(columns=['url', 'depth'])
        self.duplicated = 0
        url_list = []
        depth_list = []
        html = self.browser.page_source
        soup = bs(html, 'html.parser')
        atag = soup.find_all('a', class_='CategoryTreeLabel')
        for a in atag:
            url_list.append(a['href'])
            depth_list.append(self.depth)
        self.depth += 1
        while self.depth < self.search_depth:
            start = default_timer()
            time.sleep(30)
            expand_buttons = self._get_expand_buttons()
            time.sleep(30)
            for button in expand_buttons:
                time.sleep(.05)
                if button.is_displayed():
                    button.click()
                else:
                    continue
            end = default_timer()
            print(f'depth of {self.depth} took {str(round((end-start)/60, 2))}' +
                  ' minutes to open')
            html = self.browser.page_source
            soup = bs(html, 'html.parser')
            atag = soup.find_all('a', class_='CategoryTreeLabel')
            for a in atag:
                link = a['href']
                if link not in url_list:
                    url_list.append(a['href'])
                    depth_list.append(self.depth)
                elif link in url_list:
                    self.duplicated += 1
            self.depth += 1
        self.df = pd.DataFrame(list(zip(url_list, depth_list)),
                               columns=['url', 'depth'])
        self._convert_utf8()
        if self.save == 'csv':
            start = default_timer()
            self._save_csv()
            end = default_timer()
            print(str(round((end-start)/60, 2)) + f' minutes to save to csv')

    def _save_csv(self):
        """Save to a csv file

            ----------
            Parameters
            ----------

            Returns
            -------

        """
        # Save pages and categories to a 'seed_pages' dir
        self.df.to_csv(f'seed/{ self.category }_d{ self.depth }.csv',
                       sep='\t', encoding='utf-8', index=False)

    def _convert_utf8(self):
        """Convert the url column to utf-8 encoding
            ----------
            Parameters
            ----------

            Returns
            -------

        """
        self.df['url'] = self.df['url'].map(unquote)

    def scrape(self, category, search_depth=3, save='csv'):
        """Scrape for either categories or all categories and pages
            ----------
            Parameters
            ----------

            Returns
            -------

        """
        self.category = category.replace(' ', '_')
        self.search_depth = search_depth
        self.save = save
        self.browser = Firefox()
        if self.all_pages == True:
            time.sleep(1)
            self.browser.get(
                f'https://en.wikipedia.org/wiki/Special:CategoryTree?target=' +
                '{ self.category }&mode=all&namespaces=&title=Special%' +
                '3ACategoryTree')
            time.sleep(1)
            self._expand_all_categories()
        else:
            time.sleep(1)
            self.browser.get(
                f'https://en.wikipedia.org/wiki/Special:CategoryTree?target=' +
                '{ self.category }&mode=categories&namespaces=&title=Special%' +
                '3ACategoryTree')
            time.sleep(1)
            self._expand_all_categories()
