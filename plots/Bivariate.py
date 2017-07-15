import os
import pandas as pd

import seaborn as sns
import numpy as np

os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')






#EDA
titanic_train.shape
titanic_train.info()

pd.crosstab(index=titanic_train["Survived"], columns="count")
#explore bivariate relationships: categorical vs categorical 
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'])
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'], margins=True)

sns.factorplot(x="Survived", hue="Sex", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#explore bivariate relationships: categorical vs continuous 
sns.factorplot(x="Fare", row="Survived", data=titanic_train, kind="box", size=6)

sns.FacetGrid(titanic_train, row="WWEEWQ",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.boxplot, "Fare").add_legend()

#explore bivariate relationships: continuous vs continuous 
np.cov(titanic_train['SibSp'], titanic_train['Parch'])
np.corrcoef(titanic_train['SibSp'], titanic_train['Parch'])
sns.jointplot(x="SibSp", y="Parch", data=titanic_train)
