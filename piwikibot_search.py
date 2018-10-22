# https://www.mediawiki.org/wiki/Special:MyLanguage/Manual:Pywikibot
import pywikibot
from pywikibot import pagegenerators

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

for item in generator:
    print(item)