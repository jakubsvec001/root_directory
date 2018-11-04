import os
import time
import sys
import csv
import pandas as pd
from selenium.webdriver import Firefox
from timeit import default_timer
from urllib.parse import unquote


def get_expand_buttons(browser):
    """Return a list of expand buttons to click on."""
    return browser.find_elements_by_xpath("//span[@title='expand']")

def expand_all_categories(browser, category, search_depth=3, save='csv', all_pages=True):
    """Expand all categories on the page."""
    start_time = default_timer()
    expand_buttons = get_expand_buttons(browser)
    time.sleep(3)
    depth = 0
    print('num expand buttons ', len(expand_buttons))
    while depth < search_depth:
        start = default_timer()
        for button in expand_buttons:
            if button.is_displayed():
                button.click()
        time.sleep(3)
        expand_buttons = get_expand_buttons(browser)
        end = default_timer()
        print(f'depth of {depth} took {str(round((end-start)/60, 2))} minutes to open')
        depth += 1
    if save == 'csv':
        start = default_timer()
        if all_pages:
            with open(f'seed_pages/all_pages_{category}_d{depth}.csv', 'w') as out:
                writer = csv.writer(out, lineterminator='\n')
                for a in browser.find_elements_by_xpath('.//a'):
                    writer.writerow([a.get_attribute('href')])
        else:
            with open(f'seed_categories/cat_pages_{category}_d{depth}.csv', 'w') as out:
                writer = csv.writer(out, lineterminator='\n')
                for a in browser.find_elements_by_xpath('.//a'):
                    writer.writerow([a.get_attribute('href')])
        end = default_timer()
        print(str(round((end-start)/60, 2)) + f' minutes to save to csv')
    if save == False:
        pass
    end_time = default_timer()
    print(str(round((end_time-start_time)/60, 2)) + f' minutes to finish search')

def get_pages(category, search_depth=3, save='csv'):
    """get the links from wikipedia's hidden category tree finder"""
    browser = Firefox()
    time.sleep(0.5)
    browser.get(f'https://en.wikipedia.org/wiki/Special:CategoryTree?target={category}&mode=all&namespaces=&title=Special%3ACategoryTree')
    time.sleep(0.5)
    category = category.replace(' ','_')
    expand_all_categories(browser, category, search_depth, save='csv', all_pages=True)

def get_categories(category, search_depth=5, save='csv'):
    """gets the links from wikipedia's hidden category tree finder"""
    browser = Firefox()
    time.sleep(0.5)
    browser.get(f'https://en.wikipedia.org/wiki/Special:CategoryTree?target={category}&mode=categories&namespaces=&title=Special%3ACategoryTree')
    time.sleep(0.5)
    category = category.replace(' ','_')
    expand_all_categories(browser, category, search_depth, save='csv', all_pages=False)

def only_open(category, search_depth=5):
    browser = Firefox()
    time.sleep(0.5)
    browser.get(f'https://en.wikipedia.org/wiki/Special:CategoryTree?target={category}&mode=all&namespaces=&title=Special%3ACategoryTree')
    time.sleep(0.5)
    category = category.replace(' ','_')
    expand_all_categories(browser, category, search_depth, save=False, all_pages=False)

def clean_url(url):
    base_url = 'https://en.wikipedia.org/wiki/'
    url = unquote(url).strip('')
    return url[len(base_url):]

# def main():
#     """Open the page and expand all categories."""
#     browser = Firefox()
#     browser.get('https://en.wikipedia.org/wiki/Special:CategoryTree?target=mathematics&mode=all&namespaces=&title=Special%3ACategoryTree')
#     expand_all_categories(browser, 'physics', search_depth=3, save='csv')