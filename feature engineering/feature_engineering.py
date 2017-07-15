import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import seaborn as sns

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


               
#fare_distribution()

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
        
#process names of passengers
title_Dictionary = {
                        "Capt":       "Officer", "Col":        "Officer",
                        "Major":      "Officer", "Jonkheer":   "Royalty",
                        "Don":        "Royalty", "Sir" :       "Royalty",
                        "Dr":         "Officer", "Rev":        "Officer",
                        "the Countess":"Royalty","Dona":       "Royalty",
                        "Mme":        "Mrs", "Mlle":       "Miss",
                        "Ms":         "Mrs", "Mr" :        "Mr",
                        "Mrs" :       "Mrs", "Miss" :      "Miss",
                        "Master" :    "Master", "Lady" :      "Royalty"
}


def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

titanic_train['Title'] = titanic_train['Name'].map(extract_title)
titanic_train['Title'] = titanic_train['Title'].map(title_Dictionary)

titanic_test['Title'] = titanic_test['Name'].map(extract_title)
titanic_test['Title'] = titanic_test['Title'].map(title_Dictionary)
        


titanic_train['FamilySize'] = titanic_train.SibSp + titanic_train.Parch + 1
titanic_train['FamilyType'] = titanic_train['FamilySize'].map(size_to_type)

titanic_test['FamilySize'] = titanic_test.SibSp + titanic_test.Parch + 1
titanic_test['FamilyType'] = titanic_test['FamilySize'].map(size_to_type)

concat_df = pd.concat([titanic_train , titanic_test])

features_df  = pd.get_dummies(concat_df, columns=['Pclass', 'Sex', 'Embarked','Ticket','Title','FamilyType'])
titanic_train1 = features_df[features_df['label'] == 'train']
titanic_test1 = features_df[features_df['label'] == 'test']
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin', 'Name','Survived','label'], 1)
y_train = titanic_train['Survived']

#### adaboost...
#dt_estimator = tree.DecisionTreeClassifier(max_depth=3)
#ada_tree_estimator1 = ensemble.AdaBoostClassifier(dt_estimator, 5)
#ada_grid = {'n_estimators':[5],'learning_rate':[0.1,0.5,0.7,0.9]}
#ada_grid_estimator = model_selection.GridSearchCV(ada_tree_estimator1,ada_grid, cv=10, n_jobs=10)
#ada_grid_estimator.fit(X_train, y_train)
#ada_grid_estimator.grid_scores_
#ada_grid_estimator.best_score_
#ada_grid_estimator.score(X_train, y_train)
#best_est = ada_grid_estimator.best_estimator_
#
####Random forest
rf_estimator = ensemble.RandomForestClassifier(random_state=2017)
rf_grid = {'n_estimators':list(range(50,500,50)),'max_features':[3,4,5,6,7,8,9],'criterion':['entropy','gini']}
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator,rf_grid, cv=10, n_jobs=10)
rf_grid_estimator.fit(X_train, y_train)
rf_grid_estimator.grid_scores_
rf_grid_estimator.best_estimator_
rf_grid_estimator.best_score_
rf_grid_estimator.best_estimator_.feature_importances_
rf_grid_estimator.score(X_train, y_train)


dt = tree.DecisionTreeClassifier()
param_grid = {'max_depth':[3,4,5,6,7,8,9,10], 'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12]}
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=5)
dt_grid.fit(X_train, y_train)
dt_grid.best_params_ 
dt_grid.grid_scores_
featureImp = pd.DataFrame({'cols' : X_train.columns, 'Imp_ab' :list(dt_grid.best_estimator_.feature_importances_)})
featureImp.sort_values(by='Imp_ab',ascending=False,inplace = True)
c =dt_grid.best_estimator_.feature_importances_
dt_grid.best_score_
dt_grid.score(X_train, y_train)

titanic_test.shape

#fare_distribution_test()
titanic_test.describe()
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
titanic_test['Survived'] = dt_grid.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)

sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)

sns.FacetGrid(titanic_train, row="Survived", col="Pclass").map(sns.countplot, "Embarked")


def find_childern(x):
    print("Ticket number",x)
    count =0
    for index, row in titanic_train.iterrows():
        if(row['Age'] < 12):
            print(row['PassengerId'],":",extract_title(row['Name']))
            count = count+1
    
    return count

print("children count",find_childern(1))



def assign_children(x):
    if(x<12):
        return "children"
    elif(x>12):
        return "elder"

titanic_train['Age_type'] = titanic_train['Age'].map(assign_children)


def Age_type():
    print("Ticket number")
    count =0
    for index, row in titanic_train.iterrows():
        if(row['Age_type'] != "children" and row['Age_type'] !=  "elder"):
            if(row['Title'] !=('Master' or 'Miss')):
                 if((row['Title'] =='Miss' and row['FamilySize']-1==0) or 
                    (row['Title'] =='Master') or (row['Title'] !='Miss') ):
                      #print(row['Title'],'elder') 
                       titanic_train.loc[index, 'Age_type'] =  'elder'
                 elif(row['Title'] =='Mr' or row['Title'] =='Mrs'):
                      titanic_train.loc[index, 'Age_type'] =  'elder'
                     # print(row['Title'],'elder')
                 else:
                      titanic_train.loc[index, 'Age_type'] =  'children'
                     # count = count+1
                    #  print(row['PassengerId'])
            else:
                 print(row['Title'],'children')
            
    return count


print(Age_type())


def Age_type_test():
    print("Ticket number")
    count =0
    for index, row in titanic_test.iterrows():
        if(row['Age_type'] != "children" and row['Age_type'] !=  "elder"):
            if(row['Title'] !=('Master' or 'Miss')):
                 if((row['Title'] =='Miss' and row['FamilySize']-1==0) or 
                    (row['Title'] !='Miss') ):
                      #print(row['Title'],'elder') 
                       titanic_test.loc[index, 'Age_type'] =  'elder'
                 elif(row['Title'] =='Mr' or row['Title'] =='Mrs'):
                      titanic_test.loc[index, 'Age_type'] =  'elder'
                     # print(row['Title'],'elder')
                 elif(row['Title'] =='Master'):
                       titanic_test.loc[index, 'Age_type'] =  'children'
                 else:
                      titanic_test.loc[index, 'Age_type'] =  'children'
                     # count = count+1
                    #  print(row['PassengerId'])
            else:
                 print(row['Title'],'children')
            
    return count

print(Age_type_test())
titanic_test['Age_type'] = titanic_test['Age'].map(assign_children)

sns.factorplot(x="Survived", hue="Age_type", data=titanic_train, kind="count", size=6)
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Age_type'])

