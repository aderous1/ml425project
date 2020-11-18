import pandas as pd
import numpy as np
import time
import datetime
from math import isnan

now = datetime.datetime.now()
#
# Preprocessing
#
# Read in Congress demographics and convert non-numbers to numbers
congress = pd.read_csv("data/Congress_c.csv")

bday = 'birthday'
for i in range(len(congress.loc[:,bday])):
    dt = datetime.datetime.strptime(congress.loc[i,bday], '%m/%d/%Y')
    age = now - dt
    # Age is stored in number of days
    congress.loc[i,bday] = age.days

party = 'party'
parties_dict = {'D' : -1,'I' : 0,'R' : 1}
for i in range(len(congress.loc[:,party])):
    congress.loc[i,party] = parties_dict[congress.loc[i,party][0]]

chamber = 'chamber'
chamber_dict = {'r' : -0.5,'s' : 0.5}
for i in range(len(congress.loc[:,chamber])):
    congress.loc[i,chamber] = chamber_dict[congress.loc[i,chamber][0]]

gender = 'gender'
gender_dict = {'M' : -0.5,'F' : 0.5}
for i in range(len(congress.loc[:,gender])):
    congress.loc[i,gender] = gender_dict[congress.loc[i,gender]]

race = 'raceEthnicity'
race_dict = {'A' : -1,'P' : -1,'B' : -0.5,'H' : 0,'N' : 0.5,'W' : 1}
for i in range(len(congress.loc[:,race])):
    if(not(pd.isnull(congress.loc[i,race]))):
        congress.loc[i,race] = race_dict[congress.loc[i,race][0]]
    else:
        # No race listed for some, especially non-US-State reps
        congress.loc[i,race] = 0

pre2018 = 'pre2018incumbent'
pre_dict = {'L' : -1,'N' : -0.33,'O' : 0.33,'V' : 0.33,'W' : 1}
for i in range(len(congress.loc[:,pre2018])):
    if(not(pd.isnull(congress.loc[i,pre2018]))):
        congress.loc[i,pre2018] = pre_dict[congress.loc[i,pre2018][0]]
    else:
        # No seat listed for some
        congress.loc[i,pre2018] = 0

pop = 'citizenPopulation'
for i in range(len(congress.loc[:,pop])):
    if(pd.isnull(congress.loc[i,pop])):
        congress.loc[i,pop] = 0

inc = 'medianHouseholdIncome'
for i in range(len(congress.loc[:,inc])):
    if(pd.isnull(congress.loc[i,inc])):
        congress.loc[i,inc] = 0

congress.to_csv('data/Congres_cNUMS.csv')
print('Done')
