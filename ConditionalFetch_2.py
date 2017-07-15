import os
import pandas as pd

os.getcwd()

os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
type(titanic_test)
type(titanic_train)
## Returns rows,coloumns
shape = titanic_train.shape
shape[0]
titanic_train.info()
## Only for numerical coloumns
titanic_train.describe()




##slice rows of a data frame.
titanic_train[0:3]
titanic_train[4:10]

## Slicing coloumns of the data frame.
titanic_train['Sex']
titanic_train[['Survived','Sex']]

## slicing both rows and coloumns
titanic_train[['Survived','Sex']]

## slice on row and coloumn access with indexes only
titanic_train.iloc[0:3,0:2]

## slice based row and coloumn access with coloumn names.
titanic_train.loc[0:0,['Survived','Sex']]

titanic_test['Survived'] =0
             
# To explore the each container.           
titanic_test.groupby('Survived').size
 titanic_test[['Survived']]
titanic_test.to_csv("submission.csv",columns=['PassengersId']) 

titanic_test.groupby('Survived').size
                    

titanic_train.groupby(['Sex','Survived']).size()
titanic_submission.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)



#9246808688


####
## Conditional slicing 

titanic_test.loc[titanic_test.Sex =='female','Survived'] =1
titanic_test[['Survived','Sex']]               
titanic_train[titanic_train.Sex=='female'].groupby(['Pclass','Sex','Survived','Embarked']).size()

titanic_train[titanic_train.Sex=='female'].groupby(['Pclass','Sex','Survived']).size()


titanic_train[titanic_train.Parch >0].groupby(['Pclass','Embarked','Survived']).size()  

titanic_genderSubmission = pd.read_csv('gender_submission.csv')

titanic_genderSubmission.loc[titanic_test.Sex =='female' ,'Survived'] =1
titanic_genderSubmission.to_csv("gender_submission.csv", columns=['PassengerId','Survived'], index=False)                      
titanic_test[['Survived','Sex','Pclass']]    
titanic_genderSubmission.groupby(['Survived']).size()  
titanic_genderSubmission.groupby(['Sex']).size()  
titanic_genderSubmission[titanic_test.Sex=='female'].groupby(['Sex','Survived']).size()  
titanic_test.groupby(['Sex']).size()                 
                    


## Version 3
titanic_train[titanic_train.Sex=='male'].groupby(['Pclass','Survived']).size()
titanic_train.groupby(['Survived','Sex']).size()

titanic_train[titanic_train.Sex=='male'].groupby(['Pclass','Embarked','Survived']).size()

titanic_train[(titanic_train.Sex =='male')].groupby(['Survived','Pclass','Parch']).size()

## 3 dimensional analysisss..........

titanic_train[(titanic_train.Sex=='female') & (titanic_train.Pclass==1) ].groupby(['Embarked','Survived']).size()
females = titanic_train[titanic_train.Sex=='female']
type(females)
females.groupby(['Embarked','Pclass','Survived']).size()

## Solution2
titanic_train[titanic_train.Sex=='female'].groupby(['Embarked','Survived','Pclass']).size()
titanic_genderSubmission.loc[(titanic_test.Sex =='female') & (titanic_test.Embarked =='S') & (titanic_test.Pclass ==3) ,'Survived'] =0
titanic_genderSubmission.to_csv("gender_submission.csv", columns=['PassengerId','Survived'], index=False)  



titanic_train[titanic_train.Sex=='female'].groupby(['Survived','Pclass']).size()