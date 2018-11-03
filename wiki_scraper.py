import os
import time
import sys
import pandas as pd
from selenium.webdriver import Firefox
from timeit import default_timer

def get_expand_buttons(browser):
    """Return a list of expand buttons to click on."""
    return browser.find_elements_by_xpath("//span[@title='expand']")

def expand_all_categories(browser, category, search_depth=3, save='csv', all_cats_pages=False):
    """Expand all categories on the page."""
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
        lst = []
        for a in browser.find_elements_by_xpath('.//a'):
            lst.append(a.get_attribute('href'))
        links = pd.DataFrame(lst)
        links = links.iloc[7:-32,:]
        if all_cats_pages:
            links.to_csv(f'seed_data/cat_only_{category}_d{depth}_{len(links)}.csv', sep=',', encoding='utf-8', header=None, index=False)
        else:
            links.to_csv(f'seed_categories/all_pages{category}_d{depth}_{len(links)}.csv', sep=',', encoding='utf-8', header=None, index=False)
        end = default_timer()
        print(str(round((end-start)/60, 2)) + f' minutes to save to csv {len(links)} hrefs')

def get_links(category, search_depth=3, save='csv'):
    """get the links from wikipedia's hidden category tree finder"""
    browser = Firefox()
    time.sleep(0.5)
    browser.get(f'https://en.wikipedia.org/wiki/Special:CategoryTree?target={category}&mode=all&namespaces=&title=Special%3ACategoryTree')
    time.sleep(0.5)
    category = category.replace(' ','_')
    expand_all_categories(browser, category, search_depth, save='csv', all_cats_pages=True)

def get_categories(category, search_depth=5, save='csv'):
    """
    category= string
    search_depth = 3
    save = 'csv'
     
    gets the links from wikipedia's hidden category tree finder
    """
    browser = Firefox()
    time.sleep(0.5)
    browser.get(f'https://en.wikipedia.org/wiki/Special:CategoryTree?target={category}&mode=categories&namespaces=&title=Special%3ACategoryTree')
    time.sleep(0.5)
    category = category.replace(' ','_')
    expand_all_categories(browser, category, search_depth, save='csv')


# def main():
#     """Open the page and expand all categories."""
#     browser = Firefox()
#     browser.get('https://en.wikipedia.org/wiki/Special:CategoryTree?target=mathematics&mode=all&namespaces=&title=Special%3ACategoryTree')
#     expand_all_categories(browser, 'physics', search_depth=3, save='csv')