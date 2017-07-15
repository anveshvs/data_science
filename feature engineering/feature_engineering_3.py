import os
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
import seaborn as sns
from sklearn import preprocessing
import numpy as np
from sklearn import tree
os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
titanic_train.shape
titanic_train.info()


def numberof_common_tickets(x):
    print("Ticket number",x)
    count =0
    for index, row in titanic_train.iterrows():
        if(row['Ticket'] == str(x)):
            count = count+1
    
    return count
    

def fare_distribution():
    num =0
    diction = {}
    for index, row in titanic_train.iterrows():
        fare =   row['Fare']
        if(fare ==0):
           continue
        if(row['Ticket'] in diction):
           print("Ticket number",row['Ticket'])
           titanic_train.loc[index, 'Fare'] = diction.get(str(row['Ticket']))
           print("Distributed fare",diction.get(str(row['Ticket'])))
           continue
           
        num = numberof_common_tickets(row['Ticket'])
        print("Count of ",row['Ticket'],":",num)
        print('Fare: ',fare/num)
        titanic_train.loc[index, 'Fare'] =  fare/num
        diction[ str(row['Ticket'])]=fare/num
               
fare_distribution()

def remove_nan_train(x):
    print('called')
    #count=0
    for index, row in titanic_train.iterrows():
         #count =count+1
         #print(type(row['Fare']))
         if(str(row[x])== "nan"):
                print(x,"nan found at",index, 'and removed')
                titanic_train.loc[index, x] =  titanic_train[x].mean()
               # print(row['PassengerId'])


def size_to_type(x):
    if(x == 1): 
        return 'Single'
    elif(x >= 2 and x <= 3): 
        return 'Small'
    elif(x >= 3 and x <= 4):
        return 'Medium'
    elif(x >= 4 and x <=6 ):
        return 'Big'
    else: 
        return 'Large'
        
titanic_train['FamilySize'] = titanic_train.SibSp + titanic_train.Parch + 1
titanic_train['FamilyType'] = titanic_train['FamilySize'].map(size_to_type)

#sns.distplot(titanic_train['FamilySize'])

#sns.distplot(titanic_train['Age'])
remove_nan_train("Age")



sns.FacetGrid(titanic_train, row="Survived", col="Pclass").map(sns.countplot, "FamilyType")
#sns.factorplot(x="Pclass", hue="FamilySize", data=titanic_train, kind="count", size=6)

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked','FamilyType'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

dt_estimator = tree.DecisionTreeClassifier(max_depth=4)
ada_tree_estimator1 = ensemble.AdaBoostClassifier(dt_estimator, 5)
ada_grid = {'n_estimators':[5,10,15,20,25,30,35,40,45],'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
ada_grid_estimator = model_selection.GridSearchCV(ada_tree_estimator1,ada_grid, cv=30, n_jobs=10)
ada_grid_estimator.fit(X_train, y_train)
ada_grid_estimator.grid_scores_
ada_grid_estimator.best_score_
ada_grid_estimator.best_params_
best_est = ada_grid_estimator.best_estimator_
ada_grid_estimator.score(X_train, y_train)



## Feature engineering for test:

def numberof_common_tickets_test(x):
    print("Ticket number",x)
    count =0
    for index, row in titanic_test.iterrows():
        if(row['Ticket'] == str(x)):
            count = count+1
    
    return count
    

def fare_distribution_test():
    num =0
    diction = {}
    for index, row in titanic_test.iterrows():
        fare =   row['Fare']
        if(fare ==0):
           continue
        if(row['Ticket'] in diction):
           print("Ticket number",row['Ticket'])
           titanic_test.loc[index, 'Fare'] = diction.get(str(row['Ticket']))
           print("Distributed fare",diction.get(str(row['Ticket'])))
           continue
           
        num = numberof_common_tickets_test(row['Ticket'])
        print("Count of ",row['Ticket'],":",num)
        print('Fare: ',fare/num)
        titanic_test.loc[index, 'Fare'] =  fare/num
        diction[ str(row['Ticket'])]=fare/num
               
fare_distribution_test()


titanic_test['FamilySize'] = titanic_test.SibSp + titanic_test.Parch + 1
titanic_test['FamilyType'] = titanic_test['FamilySize'].map(size_to_type)  





titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked','FamilyType'])
titanic_test1.shape
titanic_test1.info()
titanic_test1.head(6)

def remove_nan_test(x):
    print('called')
    #count=0
    for index, row in titanic_test.iterrows():
         #count =count+1
         #print(type(row['Fare']))
         if(str(row[x])== "nan"):
                print(x,"nan found at",index, 'and removed')
                titanic_test.loc[index, x] =  titanic_test[x].mean()
               # print(row['PassengerId'])
remove_nan_test("Age")

X_testo = titanic_test1.drop(['PassengerId','Cabin','Ticket', 'Name'], 1)

def remove_nan(x):
    #count=0
    for index, row in X_testo.iterrows():
         #count =count+1
         #print(type(row['Fare']))
         if(str(row[x])== "nan"):
                print(x,"nan found at",index, 'and removed')
                X_testo.loc[index, x] =  X_testo[x].mean()
               # print(row['PassengerId'])
        
remove_nan('Fare')



titanic_test['Survived'] = ada_grid_estimator.predict(X_testo)
titanic_test.to_csv("gender_submission.csv", columns=['PassengerId','Survived'], index=False)
