#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jeffreybruggeman
"""

import pandas as pd
# import data
X_train = pd.read_csv("X_train.csv", sep=',')
X_test = pd.read_csv("X_test.csv", sep=',')
y_train = pd.read_csv("y_train.csv", sep=',')
y_test = pd.read_csv("y_test.csv", sep=',')

# create random forest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10000, random_state=10, n_jobs=-1)
forest.fit(X_train, y_train)
#save impurity reduction information for each attribute into this array
importances = forest.feature_importances_
# create this DF to hold the importances
assessment = pd.DataFrame(columns=X_train.columns)
assessment.loc[1] = importances
# sort the DF based on the importances values for each feature (so the first column will hold the largest impurity and so on)
assessment.sort_values(by=1,axis=1,ascending=False,inplace=True)
# print out this DF using the print regex given in Dr Angryk's slides
k=0
for (attr, Series) in assessment.iteritems():
    print('%2d) %-*s %f' % (k+1, 20, attr, assessment[attr]))
    k=k+1
# saved CSV for use in task 5
assessment.to_csv("assessment.csv", encoding='utf-8', index=False)

