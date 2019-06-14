#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jeffreybruggeman
"""

import pandas as pd
import numpy as np

#import file
df = pd.read_csv("CancerData.csv", sep=',')
#Learn class identifier
df.info()
#Class mapping and normaliaztion
class_map = {label:value for value,label in enumerate(np.unique(df['Class']))}
print(class_map)

for (name, Series) in df.iteritems():
    if (name=='Class'):
        #map class
        df[name] = df[name].map(class_map)
    else:
        # normalize data on range 0,2
        col_min=df[name].min()
        col_max=df[name].max()
        for x in range(int(len(df[name]))):
            df[name].at[x]=( (df[name].at[x]-col_min)/(col_max-col_min) ) * 2
# save csv
df.to_csv('cancerNormalized.csv', encoding='utf-8', index=False)

# Made a splom to visualize the data for task 5
# =============================================================================
# import seaborn as sns
# sns.set(style='ticks')
# sns.pairplot(df, hue='Class')
# =============================================================================
