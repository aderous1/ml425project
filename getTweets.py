import twitter
import json

f = open('twitterkeys.json')
keys = json.load(f)
f.close()

key = keys['key']
secretKey = keys['secretKey']
bearer = keys['bearer']
accessKey = keys['access']
accessSecret = keys['accessSecret']

api = twitter.Api(consumer_key=key,consumer_secret=secretKey,access_token_key=accessKey,access_token_secret=accessSecret)

search = api.GetUserTimeline(screen_name='RepKevinYoder', )