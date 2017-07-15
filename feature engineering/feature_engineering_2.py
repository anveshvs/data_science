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
           print('bye')
           print("Ticket number",row['Ticket'])
           titanic_test.loc[index, 'Fare'] = diction.get(str(row['Ticket']))
           print("Distributed fare",diction.get(str(row['Ticket'])))
           continue
        print('hi')
        num = numberof_common_tickets_test(row['Ticket'])
        print("Count of ",row['Ticket'],":",num)
        print('Fare: ',fare/num)
        titanic_test.loc[index, 'Fare'] =  fare/num
        diction[ str(row['Ticket'])]=fare/num
               
               

fare_distribution_test()

def remove_nan():
    #count=0
    for index, row in X_test.iterrows():
         #count =count+1
         #print(type(row['Fare']))
         if(str(row['Fare'])== "nan"):
                print("nan found at",index, 'and removed')
                X_test.loc[index, 'Fare'] =  X_test['Fare'].mean()
               # print(row['PassengerId'])
        


titanic_train1 = pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
X_train = titanic_train1.drop(['PassengerId','Age','Name','Survived','Ticket','Cabin'],1)
y_train = titanic_train['Survived']


titanic_test1 = pd.get_dummies(titanic_test,columns=['Pclass','Sex','Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Name','Ticket','Cabin'],1)


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

remove_nan()



titanic_test['Survived'] = gridSearch_val.predict(X_test)
titanic_test.to_csv("gender_submission.csv", columns=['PassengerId','Survived'], index=False)
