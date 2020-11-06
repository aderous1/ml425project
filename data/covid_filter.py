import pandas as pd
import re
from os import path

if re.search('virus', 'CHINA VIRUS', re.IGNORECASE):
    print(True)
wordbank = ['corona','coronavirus','covid','covid19','covid-19','virus',\
            'pandemic','epidemic','lockdown','testing','cases',\
            'hospitalization','related deaths','mask','cdc','WHO',\
            'conspiracy','sick','hospital','health']

congress = pd.read_csv("congressTwits.csv")
handles = congress.loc[:,'TweetCongress']
index = 0
for handle in handles:
    if(isinstance(handle,float)):
        congress.drop(index)
        continue
    print('Filtering tweets for '+handle)
    f = 'ml425project-176192ff8575f26cc59cf8b1dec2543a8c47f80c/data/tweets_1000/'+\
        handle+'.csv'
    if(not path.exists(f)):
        congress.drop(index)
        continue
    if(handle == 'RepRichHudson'):
        congress.drop(index)
        continue
    df = pd.read_csv(f,header=None)
    covid_tweets = pd.DataFrame()
    covid_tweets['Tweet'] = []
    tweets = df.iloc[:,3]
    for tweet in tweets:
        for word in wordbank:
            if re.search(word,tweet,re.IGNORECASE):
                covid_tweets = covid_tweets.append({'Tweet':tweet},ignore_index = True)
                break
    n = len(covid_tweets)
    print(handle+' had '+str(n)+' covid related tweets')
    print()
    f = 'ml425project-176192ff8575f26cc59cf8b1dec2543a8c47f80c/data/tweets_covid/'+\
        handle+'_c.csv'
    covid_tweets.to_csv(f)
    index += 1

congress.to_csv('Congress_c.csv')
