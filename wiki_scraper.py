from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client.oleg

# get the raw html 
url = "http://www.crummy.com/software/BeautifulSoup/bs4/doc/#"
import urllib2
html = urllib2.urlopen(url).read()
html[:100]

#'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n  "http://www.w3.org/TR/xhtml1/DTD/xh'

# store the <key:value> -> <url:html> into mongo for later use
db.tikhonov.insert({"url":url, "html":html})
ObjectId('532e6904866cd3431a90c618')
# retrieve the stored html by search the url
record = db.tikhonov.find_one({"url":url})
record['url']

# u'http://www.crummy.com/software/BeautifulSoup/bs4/doc/#'

# turn html txt into soup and start parsing
from bs4 import BeautifulSoup
soup = BeautifulSoup(record['html'])
soup.find("h1").text

# u'Beautiful Soup Documentation\xb6'