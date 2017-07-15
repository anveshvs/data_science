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
dt = tree.DecisionTreeClassifier()
bag_tree_estimator1 = ensemble.BaggingClassifier(dt, 5)
scores = model_selection.cross_val_score(bag_tree_estimator1,X_train,y_train,cv =10,verbose=1)
dt.fit(X_train,y_train)
scores.mean()



#OOB  for bagged tree ensemble
bag_tree_estimator2 = ensemble.BaggingClassifier(dt, 5, oob_score=True)
bag_tree_estimator2.fit(X_train, y_train)
bag_tree_estimator2.oob_score_


# Tunning bagged ensemble tree....

dt = tree.DecisionTreeClassifier(random_state=1017)
bt_bag_estimator = ensemble.BaggingClassifier(dt,n_jobs=1)
bt_grid = {'max_features':[1,2,3,4,5,6,7,8,9],'n_estimators':list(range(100,500,100))}
bag_grid_estimator = model_selection.GridSearchCV(bt_bag_estimator,bt_grid, cv=10, n_jobs=10)
bag_grid_estimator.fit(X_train, y_train)
bag_grid_estimator.grid_scores_
bag_grid_estimator.best_score_
bag_grid_estimator.estimator
bag_grid_estimator.best_params_
bag_grid_estimator.score(X_train, y_train)

titanic_test['Survived'] = bag_grid_estimator.predict(X_test)
titanic_test.to_csv("gender_submission.csv", columns=['PassengerId','Survived'], index=False)
