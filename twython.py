import twython
import json

f = open('twitterkeys.json')
keys = json.load(f)
f.close()

key = keys['key']
secretKey = keys['secretKey']
bearer = keys['bearer']
accessKey = keys['access']
accessSecret = keys['accessSecret']

twitter = twython. (key, access_token=secretKey)

twython.
twitter.search(q='(corona OR virus OR covid OR covid19) (from:realDonaldTrump) -filter:replies')

