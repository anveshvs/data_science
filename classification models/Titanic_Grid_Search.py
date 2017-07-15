import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import seaborn as sns
from sklearn import preprocessing
import numpy as np

os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
titanic_train.shape
titanic_train.info()



titanic_train.apply(lambda x : sum(x.isnull()))


titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S'

#pre-process Age
imputer = preprocessing.Imputer()
type(titanic_train[['Age']])
titanic_train[['Age']] = imputer.fit_transform(titanic_train[['Age']])
sns.distplot(titanic_train['Age'])
sns.factorplot(x="Age", row="Survived", data=titanic_train, kind="box", size=6)
sns.factorplot(x="Age", row="Survived", data=titanic_train, kind="box", size=6)
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Age").add_legend()


#create family size feature
def size_to_type(x):

    if(x == 1): 
        return 'Single'
    elif(x >= 2 and x <= 4): 
        return 'Small'
    else: 
        return 'Large'
    
    
#create family size feature
def size_to_type(x):

    
    
    
    
titanic_train['FamilySize'] = titanic_train.SibSp + titanic_train.Parch + 1
titanic_train['FamilyType'] = titanic_train['FamilySize'].map(size_to_type)

type(titanic_train)

titanic_train1 = pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
X_train = titanic_train1.drop(['PassengerId','Age','Name','Survived','Ticket','Cabin','Fare'],1)
y_train = titanic_train['Survived']
X_train.apply(np.mean)

titanic_test1 = pd.get_dummies(titanic_test,columns=['Pclass','Sex','Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Name','Ticket','Cabin','Fare'],1)


dt = tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)
param_grid = {'min_samples_leaf':[3,4,5,6,7,8,9],'min_samples_split':[3,4,5,6,7,8], 'max_depth':[3,4,5,6,7] ,'max_features':[3,4,5,6,7,8,9],'criterion':['entropy','gini']}
gridSearch_val = model_selection.GridSearchCV(dt,param_grid,cv=10,refit = True,n_jobs= 10)
gridSearch_val.fit(X_train,y_train)
gridSearch_val.grid_scores_
gridSearch_val.best_params_
gridSearch_val.best_score_
gridSearch_val.best_estimator_
# Finding train error.
gridSearch_val.score(X_train, y_train)


titanic_test['Survived'] = gridSearch_val.predict(X_test)
titanic_test.to_csv("gender_submission.csv", columns=['PassengerId','Survived'], index=False)
