import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-white')
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['font.family'] = "serif"
#
# Preprocessing
#
# Read in Congress demographics and convert non-numbers to numbers
congress = pd.read_csv("data/Congress_cNUMS.csv")
features = ['party','birthday','chamber','gender','raceEthnicity','pre2018incumbent','citizenPopulation','medianHouseholdIncome']
tweets = pd.read_csv("data/rankings.csv")
congress.insert(14,"reliability",np.zeros(len(congress)),True)
congress.insert(15,"tweets",np.zeros(len(congress)),True)
congress = congress.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
# Make all handles lower case
for i in range(len(congress.loc[:,'TweetCongress'])):
    congress.loc[i,'TweetCongress'] = congress.loc[i,'TweetCongress'].lower()
print(congress)
# Tally reliability
for i in range(len(tweets)):
    handle = tweets.iloc[i]['author']
    index = congress.index[congress['TweetCongress'] == handle.lower()]
    index = index[0]
    congress.iat[index,12] = 1 - tweets.iloc[i]['ratio']
    congress.iat[index,13] = tweets.iloc[i]['one'] + tweets.iloc[i]['zero']

congress.to_csv('data/CONGRESS_RANKED.csv')
