import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import decomposition
import seaborn as sns
from sklearn import ensemble
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper

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
sns.factorplot(x="Survived", hue="Pclass", data=concat_df, kind="count", size=6)


titanic_all  = pd.get_dummies(concat_df, columns=['Sex', 'Embarked','Ticket','label'])


titanic_all1 = titanic_all.drop(['PassengerId','Age','Cabin', 'Name','Survived'], 1)

#titanic_all1.Embarked[titanic_all1['Embarked'].isnull()] = 'S'
titanic_all1.Fare[titanic_all1['Fare'].isnull()] = titanic_all1['Fare'].mean()

titanic_train1 = titanic_all1[titanic_all1['label_train'] == 1]
titanic_train1 = titanic_train1.drop(['label_train'], 1)

titanic_test1 = titanic_all1[titanic_all1['label_test'] ==1]
titanic_test1 = titanic_test1.drop(['label_test'], 1)

mapper = DataFrameMapper([(titanic_train1.columns, preprocessing.StandardScaler())])
scaled_features = mapper.fit_transform(titanic_train1)
type(scaled_features)
titanic_all_999 = pd.DataFrame(scaled_features, columns=titanic_train1.columns)



pca = decomposition.PCA(n_components=680)
pca.fit(titanic_all_999)
explainedVariance = pca.explained_variance_
varianceRatio = pca.explained_variance_ratio_
varianceCumSum = pca.explained_variance_ratio_.cumsum()
titanic_train2 = pd.DataFrame(pca.transform(titanic_all_999))


#X_train = titanic_train1.drop(['PassengerId','Age','Cabin', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#rf_estimator = ensemble.RandomForestClassifier(random_state=2017)
#rf_grid = {'n_estimators':list(range(50,500,50)),'max_features':[3,4,5,6,7,8,9],'criterion':['entropy','gini']}
#rf_grid_estimator = model_selection.GridSearchCV(rf_estimator,rf_grid, cv=10, n_jobs=10)
#rf_grid_estimator.fit(titanic_train2, y_train)
##featureImp = pd.DataFrame({'cols' : X_train.columns, 'Imp_ab' :list(rf_grid_estimator.best_estimator_.feature_importances_)})
##featureImp.sort_values(by='Imp_ab',ascending=False,inplace = True)
#rf_grid_estimator.grid_scores_
#rf_grid_estimator.best_estimator_
#print("CvScore",rf_grid_estimator.best_score_)
#rf_grid_estimator.best_estimator_.feature_importances_
#print("Train accuracy",rf_grid_estimator.score(titanic_train2, y_train))




dt = tree.DecisionTreeClassifier()
param_grid = {'max_depth':[3,4,5,6,7,8,9,10], 'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12]}
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=5)
dt_grid.fit(titanic_train2, y_train)
dt_grid.grid_scores_
dt_grid.best_estimator_
dt_grid.best_score_
dt_grid.score(titanic_train2, y_train)

titanic_test2 = pd.DataFrame(pca.transform(titanic_test1))

titanic_test['Survived'] = dt_grid.predict(titanic_test2)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)
