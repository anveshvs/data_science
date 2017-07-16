import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import decomposition
import seaborn as sns
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor

#returns current working directory
os.getcwd()
#changes working directory
os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv("train.csv")
titanic_train['label'] = 'train'
titanic_test = pd.read_csv("test.csv")
titanic_test['label'] = 'test'

titanic_train.info()



concat_df = pd.concat([titanic_train , titanic_test])
concat_df['Pclass'] = concat_df.Pclass.astype('category',categories = [3,2,1],ordered = True)

##Ploting
pd.crosstab(index=concat_df['Survived'], columns=concat_df['Pclass'])
#sns.factorplot(x="Survived", hue="Age", data=concat_df, kind="count", size=6)


titanic_all  = pd.get_dummies(concat_df, columns=['Sex', 'Embarked','Ticket','label'])

titanic_all1 = titanic_all.drop(['Cabin', 'Name','Survived'], 1)


#titanic_all1.Embarked[titanic_all1['Embarked'].isnull()] = 'S'
titanic_all1.Fare[titanic_all1['Fare'].isnull()] = titanic_all1['Fare'].mean()
#Age featuring :


ageTestDf = titanic_all1['Age'].isnull()
age_test = titanic_all1[ageTestDf]

X_test_age = age_test.drop(['Age'], 1)
y_test_age = age_test['Age']

ageTrainDf = titanic_all1['Age'].notnull()
age_train = titanic_all1[ageTrainDf]

X_train_age = age_train.drop(['Age'], 1)
y_train_age = age_train['Age']

regr_1 = DecisionTreeRegressor(max_depth=2)

regr_1.fit(X_train_age, y_train_age)






titanic_train1 = titanic_all1[titanic_all1['label_train'] == 1]
titanic_test1 = titanic_all1[titanic_all1['label_test'] ==1]

titanic_train1 = titanic_train1.drop(['label_train'], 1)
titanic_test1 = titanic_test1.drop(['label_test','Survived'], 1)

#X_train = titanic_train1.drop(['PassengerId','Age','Cabin', 'Name','Survived'], 1)
y_train = titanic_train['Survived']


