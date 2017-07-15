import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble

#returns current working directory
os.getcwd()
#changes working directory
os.chdir('C:/Projects/DataScience/Titanic_machineLearning')

titanic_train = pd.read_csv("train.csv")

#EDA
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

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier()
param_grid = {'min_samples_leaf':[3,4,5,6,7,8,9],'min_samples_split':[3,4,5,6,7,8], 'max_depth':[3,4,5,6,7] ,'max_features':[3,4,5,6,7,8,9],'criterion':['entropy','gini']}
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=5)
dt_grid.fit(X_train, y_train)
dt_grid.grid_scores_
dt_grid.best_estimator_
dt_grid.best_score_
dt_grid.score(X_train, y_train)
#rf_estimator = ensemble.RandomForestClassifier(random_state=2017)
#rf_grid = {'n_estimators':list(range(50,500,50)),'max_features':[3,4,5,6,7,8,9],'criterion':['entropy','gini']}
#rf_grid_estimator = model_selection.GridSearchCV(rf_estimator,rf_grid, cv=10, n_jobs=10)
#rf_grid_estimator.fit(X_train, y_train)
#rf_grid_estimator.grid_scores_
#rf_grid_estimator.best_estimator_
#rf_grid_estimator.best_score_
#rf_grid_estimator.best_estimator_.feature_importances_
#rf_grid_estimator.score(X_train, y_train)



titanic_test = pd.read_csv("test.csv")
titanic_test.shape

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

titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = dt_grid.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)