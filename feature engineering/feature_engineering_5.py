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

titanic_train['label'] = 'train'

titanic_test = pd.read_csv("test.csv")

titanic_test['label'] = 'test'


#EDA
titanic_train.shape
titanic_train.info()




concat_df = pd.concat([titanic_train , titanic_test])

features_df  = pd.get_dummies(concat_df, columns=['Pclass', 'Sex', 'Embarked','Ticket'])
titanic_train1 = features_df[features_df['label'] == 'train']
titanic_test1 = features_df[features_df['label'] == 'test']
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin', 'Name','Survived','label'], 1)
y_train = titanic_train['Survived']


def add_elements_toList():
    listo =[]
    for index,row in featureImp.iterrows():
        if(row['Imp_ab']!=0):
            listo.append(row['cols'])
           # print(row['cols'])
        
    return listo
        
    add_elements_toList()

X_train_new = X_train[add_elements_toList()]


#dt = tree.DecisionTreeClassifier()
##dt.fit(X_train, y_train)
#param_grid = {'max_depth':[3,4,5,6,7,8,9,10], 'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12]}
#dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=5)
#dt_grid.fit(X_train_new, y_train)
#featureImp = pd.DataFrame({'cols' : X_train.columns, 'Imp_ab' :list(dt_grid.best_estimator_.feature_importances_)})
#featureImp.sort_values(by='Imp_ab',ascending=False,inplace = True)
#dt_grid.grid_scores_
#dt_grid.best_estimator_
#dt_grid.best_score_
#dt_grid.score(X_train_new, y_train)


#dt_estimator = tree.DecisionTreeClassifier(max_depth=4)
#ada_tree_estimator1 = ensemble.AdaBoostClassifier(dt_estimator, 5)
#ada_grid = {'n_estimators':[5,10,15,20],'learning_rate':[0.4,0.5,0.6,0.7,0.8]}
#ada_grid_estimator = model_selection.GridSearchCV(ada_tree_estimator1,ada_grid, cv=30, n_jobs=10)
#ada_grid_estimator.fit(X_train_new, y_train)
#featureImp = pd.DataFrame({'cols' : X_train.columns, 'Imp_ab' :list(ada_grid_estimator.best_estimator_.feature_importances_)})
#featureImp.sort_values(by='Imp_ab',ascending=False,inplace = True)
#ada_grid_estimator.grid_scores_
#ada_grid_estimator.best_score_
#ada_grid_estimator.best_params_
#best_est = ada_grid_estimator.best_estimator_
#ada_grid_estimator.score(X_train_new, y_train)



rf_estimator = ensemble.RandomForestClassifier(random_state=2017)
rf_grid = {'n_estimators':list(range(50,500,50)),'max_features':[3,4,5,6,7,8,9],'criterion':['entropy','gini']}
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator,rf_grid, cv=10, n_jobs=10)
rf_grid_estimator.fit(X_train_new, y_train)
featureImp = pd.DataFrame({'cols' : X_train.columns, 'Imp_ab' :list(rf_grid_estimator.best_estimator_.feature_importances_)})
featureImp.sort_values(by='Imp_ab',ascending=False,inplace = True)
rf_grid_estimator.grid_scores_
rf_grid_estimator.best_estimator_
rf_grid_estimator.best_score_
rf_grid_estimator.best_estimator_.feature_importances_
rf_grid_estimator.score(X_train_new, y_train)


def add_elements_toList():
    listo =[]
    for index,row in featureImp.iterrows():
        if(row['Imp_ab']!=0):
            listo.append(row['cols'])
        
    return listo
        
    
xop = add_elements_toList

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
               
#fare_distribution_test()
titanic_test.describe
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked','Ticket'])
titanic_test1.shape
titanic_test1.info()

def remove_nan():
    #count=0
    for index, row in X_test.iterrows():
         #count =count+1
         #print(type(row['Fare']))
         if(str(row['Fare'])== "nan"):
                print("nan found at",index, 'and removed')
                X_test.loc[index, 'Fare'] =  X_test['Fare'].mean()
               # print(row['PassengerId'])

X_test = titanic_test1.drop(['PassengerId','Age','Cabin', 'Name','label','Survived'], 1)
remove_nan()
X_test_new = X_test[add_elements_toList()]
titanic_test['Survived'] = rf_grid_estimator.predict(X_test_new)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)
