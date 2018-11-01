
# Mathematics (20306 nodes) 
"""
SPARQL Query:
SELECT distinct ?item ?itemLabel ?linkTo WHERE {
  { ?item wdt:P361* wd:Q395 .}
  union
  { ?item wdt:P361/wdt:P279* wd:Q395 .}
  union
  { ?item wdt:P31/wdt:P279* wd:Q1936384 .}
  union
  { ?item wdt:P921/wdt:P279* wd:Q395 .}
  
  ?article schema:about ?item.
  
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# History (138,504 nodes)
"""
SELECT DISTINCT ?item ?article WHERE {
  { ?item wdt:P361* wd:Q309. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q309. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q309. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q309. }

  ?article schema:about ?item.

  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Biology (61,674 nodes)
"""
SELECT DISTINCT ?item ?itemLabel ?linkTo WHERE {
  { ?item wdt:P361* wd:Q420. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q420. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q420. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q420. }
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Chemistry (15,674 nodes)
"""
SELECT DISTINCT ?item ?itemLabel ?linkTo WHERE {
  { ?item wdt:P361* wd:Q2329. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q2329. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q2329. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q2329. }
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Machine Learning (5,035 nodes)
"""
SELECT DISTINCT ?item ?itemLabel ?linkTo WHERE {
  { ?item wdt:P361* wd:Q2539. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q2539. }
  #UNION
  #{ ?item (wdt:P31/wdt:P279*) wd:Q2539. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q2539. }
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Engineering / Applied Science (32,600 nodes)
"""
SELECT DISTINCT ?item ?itemLabel ?linkTo WHERE {
  { ?item wdt:P361* wd:Q11023. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q11023. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q11023. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q11023. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q11023. }
  UNION
  { ?item (wdt:P361/wdt:P279*) wd:Q28797. }
  UNION
  { ?item (wdt:P31/wdt:P279*) wd:Q28797. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q28797. }
  UNION
  { ?item (wdt:P921/wdt:P279*) wd:Q28797. }
  OPTIONAL { ?item wdt:P361 ?linkTo. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Academics / People (140,837 nodes)
"""
SELECT ?person WHERE {
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

