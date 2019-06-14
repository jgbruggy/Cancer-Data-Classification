#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jeffreybruggeman
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("CancerNormalized.csv", sep=',')

#print(df)
# Set x to columns 1-31 values, set y to Class values
X, y = df.iloc[:,1:32], df['Class'].values
# Use train test split to create 4 sets of data with a test size of 1/3. Stratify by class values (y)
# Used random_state to stabilize the random data set over multiple runs of this program
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33 ,stratify=y, random_state=0)

# saved the training and testing data to csvs
df1= pd.DataFrame(data=X_train)
df1.to_csv('X_train.csv', encoding='utf-8', index=False)
df2= pd.DataFrame(data=X_test)
df2.to_csv('X_test.csv', encoding='utf-8', index=False)
df3= pd.DataFrame(data=y_train)
df3.to_csv('y_train.csv', encoding='utf-8', index=False)
df4= pd.DataFrame(data=y_test)
df4.to_csv('y_test.csv', encoding='utf-8', index=False)

# create entropy based decision tree 
from sklearn.tree import DecisionTreeClassifier
entropyModel = pd.DataFrame(columns=['Level Limit', 'Error Rate Training', 'Error Rate Testing'])
print("\nEntropy Based Classification\n")
for height in range(1,11):
    dtree=DecisionTreeClassifier(criterion='entropy', max_depth=height, random_state=0)
    dtree=dtree.fit(X_train,y_train)
    dtree.predict(X_test)
    scoreTrain = dtree.score(X_train, y_train)
    scoreTest = dtree.score(X_test, y_test)
    # save the info generated in each loop into this DF, saving the error rate not the success rate
    entropyModel.loc[height] = [height, 1-scoreTrain, 1-scoreTest]

# view DF and simple plot
print(entropyModel)
entropyModel.pop('Level Limit')
entropyModel.plot(title='Entropy Based Classification')
plt.show()

# create gini based dicision tree
print("\nGini Based Classification\n")
giniModel = pd.DataFrame(columns=['Level Limit', 'Error Rate Training', 'Error Rate Testing'])
for height in range(1,11):
    dtree=DecisionTreeClassifier(criterion='gini', max_depth=height, random_state=0)
    dtree=dtree.fit(X_train,y_train)
    dtree.predict(X_test)
    scoreTrain = dtree.score(X_train, y_train)
    scoreTest = dtree.score(X_test, y_test)
    giniModel.loc[height] = [height, 1-scoreTrain, 1-scoreTest]

# view DF and simple plot
print(giniModel)
giniModel.pop('Level Limit')
giniModel.plot(title='Gini Based Classification')
plt.show()

# After viewing most successful iteration, saved that tree as my tree.dot
dtree= DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=0)
dtree= dtree.fit(X_train, y_train)

from sklearn.tree import export_graphviz
export_graphviz(dtree, out_file='tree.dot')