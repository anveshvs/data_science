# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 21:58:45 2017
@author: Anvesh
"""
import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble
from sklearn import decomposition

#returns current working directory
os.getcwd()
#changes working directory
os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv("train.csv")
titanic_train['label'] = 'train'
titanic_test = pd.read_csv("test.csv")
titanic_test['label'] = 'test'



concat_df = pd.concat([titanic_train , titanic_test])
titanic_all  = pd.get_dummies(concat_df, columns=['Pclass', 'Sex', 'Embarked','Ticket','label'])

titanic_all1 = titanic_all.drop(['PassengerId','Age','Cabin', 'Name','Survived'], 1)

#titanic_all1.Embarked[titanic_all1['Embarked'].isnull()] = 'S'
titanic_all1.Fare[titanic_all1['Fare'].isnull()] = titanic_all1['Fare'].mean()

titanic_train1 = titanic_all1[titanic_all1['label_train'] == 1]
titanic_test1 = titanic_all1[titanic_all1['label_test'] ==1]

titanic_train1 = titanic_train1.drop(['label_train'], 1)
titanic_test1 = titanic_test1.drop(['label_test'], 1)

                      
pca = decomposition.PCA(n_components=7)
pca.fit(titanic_train1)
explainedVariance = pca.explained_variance_
varianceRatio = pca.explained_variance_ratio_
varianceCumSum = pca.explained_variance_ratio_.cumsum()
titanic_all2 = pd.DataFrame(pca.transform(titanic_train1))





#X_train = titanic_train1.drop(['PassengerId','Age','Cabin', 'Name','Survived','label'], 1)
y_train = titanic_train['Survived']


dt = tree.DecisionTreeClassifier()
param_grid = {'max_depth':[3,4,5,6,7,8,9,10], 'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12]}
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=5)
dt_grid.fit(titanic_all2, y_train)
dt_grid.grid_scores_
dt_grid.best_estimator_
dt_grid.best_score_
dt_grid.score(titanic_all2, y_train)




#X_test = titanic_test1.drop(['PassengerId','Age','Cabin', 'Name','label','Survived'], 1)
#remove_nan()
#X_test_new = X_test[add_elements_toList()]
titanic_test2 = pd.DataFrame(pca.transform(titanic_test1))
titanic_test['Survived'] = dt_grid.predict(titanic_test2)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)