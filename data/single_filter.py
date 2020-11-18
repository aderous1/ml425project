import pandas as pd
import re
from os import path

if re.search('virus', 'CHINA VIRUS', re.IGNORECASE):
    print(True)
wordbank = ['corona','coronavirus','covid','covid19','covid-19','virus',\
            'pandemic','epidemic','lockdown','testing','cases',\
            'hospitalization','related deaths','mask','cdc','WHO',\
            'conspiracy','sick','hospital','health']


handles = ['SenRickScott']
for handle in handles:

    print('Filtering tweets for '+handle)
    f = 'tweets_1000/'+\
        handle+'.csv'
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
    f = 'tweets_covid/'+\
        handle+'_c.csv'
    covid_tweets.to_csv(f)

