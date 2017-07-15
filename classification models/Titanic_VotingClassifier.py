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

#rf_estimator = ensemble.ExtraTreesClassifier(max_features=4,max_depth=4,min_samples_split=4,min_samples_leaf=2);
#scores = model_selection.cross_val_score(rf_estimator,X_train,y_train,cv =10,verbose=1)
#rf_estimator.fit(X_train,y_train)
#print(scores.mean())


#
## Tunning bagged ensemble tree....
#
dt = tree.DecisionTreeClassifier(random_state=2017)
rf = ensemble.RandomForestClassifier(random_state=2017)
adaboost = ensemble.AdaBoostClassifier(random_state=2017)

voting_grid = dict(rf__n_estimators=list(range(100,1000,100)), rf__max_features=list(range(3,8,1)),
               dt__max_depth=list(range(3,7)),
                ada__n_estimators=list(range(100,1000,100)), ada__learning_rate=[0.1,0.3,0.5])

v_estimator1 = ensemble.VotingClassifier([('dt',dt), ('rf',rf), ('ada',adaboost)])
voting_grid_estimator = model_selection.GridSearchCV(estimator=v_estimator1, param_grid=voting_grid, cv=10,n_jobs=10)
voting_grid_estimator.fit(X_train,y_train)


voting_grid_estimator.grid_scores_
print(voting_grid_estimator.best_score_)
voting_grid_estimator.estimator
print(voting_grid_estimator.best_params_)
print(voting_grid_estimator.score(X_train, y_train))

titanic_test['Survived'] = voting_grid_estimator.predict(X_test)
titanic_test.to_csv("gender_submission.csv", columns=['PassengerId','Survived'], index=False)
