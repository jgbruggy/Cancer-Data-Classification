#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jeffreybruggeman
"""

import pandas as pd
import matplotlib.pyplot as plt


# import all data
X_train = pd.read_csv("X_train.csv", sep=',')
X_test = pd.read_csv("X_test.csv", sep=',')
y_train = pd.read_csv("y_train.csv", sep=',')
y_test = pd.read_csv("y_test.csv", sep=',')


from sklearn.neighbors import KNeighborsClassifier
# created 2 dataframes, one to hold results and one to hold best results
kF= pd.DataFrame(columns=['Weight','p-Norm', 'Neighbors', 'Training Error', 'Testing Error'])
kFBest = pd.DataFrame(columns=['Weight','p-Norm', 'Neighbors', 'Training Error', 'Testing Error'])
# weight list set to iterate over in next set of loops
weight_list= {'uniform', 'distance'}
n=0
# 3 tiers of for loops for testing many different setups: weights, p-norms, and neightbors
for knn_weight in set(weight_list):
    for norm in range(1,4):
        print(f'Using "{knn_weight}" weight and p-Norm {norm}')
        for k in range(1,21):
            # build classifier
            knn = KNeighborsClassifier(n_neighbors=k, p=norm, metric='minkowski', weights=knn_weight)
            knn.fit(X_train, y_train)
            scoreTrain = knn.score(X_train, y_train)
            scoreTest = knn.score(X_test, y_test)
            # add results to DF
            kF.loc[k]=[knn_weight,norm,k,1-scoreTrain,1-scoreTest]
            # check if this result is better than 96% the testing set
            # to record each good result to check the top scorers quickly
            # Came up with 96% after looking over the better results and
            # wanted a concise list of the best results.
            if(scoreTest >= .96):
                high_score = scoreTest
                kFBest.loc[n]=[knn_weight,norm,k,1-scoreTrain,1-scoreTest]
                n=n+1
        # print kF after building each set        
        print(kF)
        # plot each DF on a scale of neighbors to error rate for testing and training sets
        kF.iloc[:,3:5].plot(title=f'Using "{knn_weight}" weight and p-Norm {norm}')
        plt.show()

# print best result
print("\nBest Testing Results:")
print(kFBest)