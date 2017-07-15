import os
import pandas as pd

import seaborn as sns


os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')






#EDA
titanic_train.shape
titanic_train.info()

#explore univariate categorical feature
titanic_train['Survived'].describe()
pd.crosstab(index=titanic_train["Survived"], columns="count")
pd.crosstab(index=titanic_train["Pclass"], columns="count")  
pd.crosstab(index=titanic_train["Sex"],  columns="count")

#explore univariate categorical features visually
sns.countplot(x='Survived',data=titanic_train)
sns.countplot(x='Pclass',data=titanic_train)

#explore univariate continuous feature
titanic_train['Fare'].describe()
sns.boxplot(x='Fare',data=titanic_train)
sns.distplot(titanic_train['Fare'])
sns.distplot(titanic_train['Fare'], bins=20, rug=True, kde=False)
sns.distplot(titanic_train['Fare'], bins=100, kde=False)
sns.kdeplot(data=titanic_train['Fare'])
sns.kdeplot(data=titanic_train['Fare'], shade=True)
