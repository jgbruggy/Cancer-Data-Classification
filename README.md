# Cancer-Data-Classification

Various classification methods used on cancer data written in python

Files include:
cancerData - the original data file including physical cancer measurements and class identifier
Cancer data classification - pdf presenting the results of the work done in IEEE format
Task 1- code that normalizes the cancer data
Task 2- splits data into train/test sets, utilizes entropy and gini to create decision trees to calculate prediction accuracy on train/test splits, outputs best model
Task 3- uses train/test split to create knn classifier 
Task 4- uses train/test split to create random forest classifier
Task 5- 7 different classifiers utilizing previous methods but with different settings. 
      Test 1: KNN for different levels of neighbors using manhattan distance and uniform weight on a mapped class set
      Test 2: Decision tree at different heights using entropy and a mapped class set
      Test 3: KNN for different levels of neighbors using manhattan distance and uniform weight on a mapped class and clamped set
      Test 4: Decision tree at different heights using entropy on a mapped class and clamped set
      Test 5: Decision tree at different heights using entropy and random forest selected attributes on the normalized set
      Test 6: KNN for different levels of neighbors using manhattan distance and uniform weight on random forest selected attributes from the normalized set
      Test 7: KNN for different levels of neighbors using manhattan distance and uniform weight on a generated set
