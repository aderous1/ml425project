import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-white')
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['font.family'] = "serif"
#
# Preprocessing
#
# Read in Congress demographics and convert non-numbers to numbers
congress = pd.read_csv("data/CONGRESS_RANKED.csv")

#Pair-plot
'''
sns.set(context='paper',style='white',font='serif')
sns.pairplot(congress, size = 1.5)
plt.show()
'''
features = ['party','chamber','gender','birthday','raceEthnicity','pre2018incumbent','citizenPopulation','medianHouseholdIncome']
X = congress.loc[:,features]

# Comment scaling for graphing
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X)
Y = congress.loc[:,['reliability']]

# Train and score algorithm
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.5,random_state=425)
reg = LinearRegression().fit(x_train,y_train)

scores = cross_val_score(reg, x_train, y_train, scoring='r2', cv=5)
print('Score:\t{:4} += {:4}'.format(scores.mean(),scores.std()))


# Linregs plots
fig, axs = plt.subplots(4,2,figsize=(8,16),sharey=True)
fig.suptitle('Reliability v. Regression Features')
fig.tight_layout(pad=3.0)
n = len(axs[0])
index = 0
X = x_test.values
Y = y_test.values
y = Y.reshape(267)

dims = [[3,20],
        [2,20],
        [2,20],
        [20,20],
        [5,20],
        [5,20],
        [20,20],
        [20,20]]
ticks = [[-1,0,1],
         [-0.5,0.5],
         [-0.5,0.5],
         [],
         [-1,-0.5,0,0.5,1],
         [-1,-0.5,0,0.5,1],
         [],[]]
labels = [['Republican','Independent','Democrat'],
          ['House','Senate'],
          ['Male','Female'],
          [],
          ['Asian','Black','Hispanic','Native Am.','White'],
          ['Lost','No race','Open','Vacant','Won'],
          [],[]]
features[3] = 'Age (days)'
for i in range(len(axs)):
    for j in range(n):
        index = (n*i)+j
        
        heatmap, xedges, yedges = np.histogram2d(X[:,index], y, bins=dims[index])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        axs[i,j].set_title(features[index])

        axs[i,j].imshow(heatmap.T, extent=extent, origin='lower',cmap='coolwarm',
                         interpolation='nearest', aspect='auto')
        if(index != 3 and index < 6):
            axs[i,j].set_xticks(ticks[index])
            axs[i,j].set_xticklabels(labels[index])
fig.show()


