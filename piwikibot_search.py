
# Mathematics (20306 nodes) 
# Only English article math articles (786 / 20306 nodes)
"""
     SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
     { ?item wdt:P361* wd:Q395. }
     UNION
     { ?item (wdt:P361/wdt:P279*) wd:Q395. }
     UNION
     { ?item (wdt:P31/wdt:P279*) wd:Q395. }
     UNION
     { ?item (wdt:P921/wdt:P279*) wd:Q395. }
     ?sitelink ^schema:name ?article.
     ?article schema:about ?item.
     ?article schema:isPartOf <https://en.wikipedia.org/>.
     OPTIONAL { ?item wdt:P361 ?linkTo. }
     SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
     """

# Math with english wikipedia entries, plus non-article nodes 
        """
        SELECT distinct ?item ?article ?sitelink ?linkTo WHERE {
          { ?item wdt:P361* wd:Q395 .}
          union
          { ?item wdt:P361/wdt:P279* wd:Q395 .}
          union
          { ?item wdt:P31/wdt:P279* wd:Q1936384 .}
          union
          { ?item wdt:P921/wdt:P279* wd:Q395 .}
          optional {?sitelink ^schema:name ?article .
                    ?article schema:about ?item ;
                    schema:isPartOf <https://en.wikipedia.org/> .
                   }
          OPTIONAL { ?item wdt:P361 ?linkTo. }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """

# Computer Science (375 / 16,706 nodes)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q21198. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q21198. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q21198. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q21198. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Physics (402 / 1,882 nodes)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q413. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q413. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q413. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q413. }
  UNION
  { ?item wdt:P361* wd:Q658544. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q658544. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q658544. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q658544. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Biology (726 / 61,674 nodes)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q420. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q420. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q420. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q420. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Chemistry (273 / 15,674 nodes)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q2329. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q2329. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q2329. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q2329. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Machine Learning (26 / 5,035 nodes)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q2539. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q2539. }
  #UNION
  #{ ?item (wdt:P31/wdt:P279*) wd:Q2539. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q2539. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# EN wiki Medicine (266 / 3526 total res)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q11190. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q11190. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q11190. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q11190. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Business ()
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q4830453. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q4830453. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q4830453. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q4830453. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# History (87884 / 138,504 nodes)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q309. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q309. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q309. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q309. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Philosophy (585 nodes)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q5891. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q5891. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q5891. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q5891. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Arts (24,442 nodes)
"""
SELECT DISTINCT ?item ?article ?sitelink ?linkTo WHERE {
  { ?item wdt:P361* wd:Q735. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q735. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q735. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q735. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Academics / People (140,837 nodes)
"""
SELECT ?person ?article WHERE {
  { ?person wdt:P106 wd:Q901. }
  UNION
  { ?person wdt:P106 wd:Q170790. }
  UNION
  { ?person wdt:P106 wd:Q169470. }
  UNION
  { ?person wdt:P106 wd:Q864503. }
  UNION
  { ?person wdt:P106 wd:Q593644. }
  UNION
  { ?person wdt:P106 wd:Q11063. }
  UNION
  { ?person wdt:P106 wd:Q4964182. }
  UNION
  { ?person wdt:P106 wd:Q82594. }
  UNION
  { ?person wdt:P106 wd:Q201788. }
  ?sitelink ^schema:name ?article.
  ?article schema:about ?item.
  ?article schema:isPartOf <https://en.wikipedia.org/>.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
"""



# https://www.mediawiki.org/wiki/Special:MyLanguage/Manual:Pywikibot
import pywikibot
from pywikibot import pagegenerators

PYWIKIBOT_NO_USER_CONFIG=1
site = pywikibot.Site()
repo = site.data_repository()
query = """SELECT distinct ?item ?itemLabel ?linkTo WHERE {
  { ?item wdt:P361* wd:Q395 .}
  union
  { ?item wdt:P361/wdt:P279* wd:Q395 .}
  union
  { ?item wdt:P31/wdt:P279* wd:Q1936384 .}
  union
  { ?item wdt:P921/wdt:P279* wd:Q395 .}
  
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}"""
generator = pagegenerators.WikidataSPARQLPageGenerator(query, site=repo)

for i in generator:
    print(i)
# i = 0

# while i < 10:
#       item = next(generator)
#       print(item)
#       i += 1

