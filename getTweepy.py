from TwitterSearch import *
import json

f = open('twitterkeys.json')
keys = json.load(f)
f.close()

key = keys['key']
secretKey = keys['secretKey']
bearer = keys['bearer']
accessKey = keys['access']
accessSecret = keys['accessSecret']

tso = TwitterSearchOrder('RepKevinYoder')

ts = TwitterSearch(consumer_key=key,consumer_secret=secretKey,access_token=accessKey,access_token_secret=accessSecret)


auth = tweepy.OAuthHandler(key,secretKey)
auth.set_access_token(accessKey,accessSecret)

api = tweepy.API(auth)


search = api.user_timeline(screen_name='RepKevinYoder', include_rts=False, k