import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble

os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
titanic_train.shape
titanic_train.info()
type(titanic_train)

titanic_train1 = pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
X_train = titanic_train1.drop(['PassengerId','Age','Name','Survived','Ticket','Cabin','Fare'],1)
y_train = titanic_train['Survived']


titanic_test1 = pd.get_dummies(titanic_test,columns=['Pclass','Sex','Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Name','Ticket','Cabin','Fare'],1)

#cv accuracy for bagged tree ensemble

rf_estimator = ensemble.ExtraTreesClassifier(max_features=4,max_depth=4,min_samples_split=4,min_samples_leaf=2);
scores = model_selection.cross_val_score(rf_estimator,X_train,y_train,cv =10,verbose=1)
rf_estimator.fit(X_train,y_train)
print(scores.mean())


#
## Tunning bagged ensemble tree....
#
ex_grid_estimator = ensemble.ExtraTreesClassifier();
rf_grid = {'max_features':[4,5,6,7,8,9],'max_depth':[4,5,6,7,8,9,10,11,12],'min_samples_split':[2,3,4,5,6,7],'min_samples_leaf':[4,5,6,7,8,9]}
ex_grid_estimator = model_selection.GridSearchCV(ex_grid_estimator,rf_grid, cv=10, n_jobs=10)
ex_grid_estimator.fit(X_train, y_train)
ex_grid_estimator.grid_scores_
ex_grid_estimator.best_score_
ex_grid_estimator.estimator
ex_grid_estimator.best_params_
ex_grid_estimator.score(X_train, y_train)

titanic_test['Survived'] = ex_grid_estimator.predict(X_test)
titanic_test.to_csv("gender_submission.csv", columns=['PassengerId','Survived'], index=False)
