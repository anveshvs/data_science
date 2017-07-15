import os
import pandas as pd

os.getcwd()

os.chdir('C:/Projects/DataScience/Titanic_machineLearning')
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')


col1=[10,20,30]
col2=['abc','def','fgh']
col3=[0,0,0]

# Creating data frame : Dictionary is the input for then frame.
df1 = pd.DataFrame({'pid':col1,'pname':col2,'survived':col3})
print(df1.shape)
df1.info
 print(df1.head(2))
 print(df1.tail(1))
 

df1['oo'] =0 

## Droping coluomn
df2 = df1.drop('oo',1) ## If there is no assignment ,df1 itself willl loose the coloumn.If there is assignment ..
   
  ## Filtering frames row by row depending on the conditin.
  df1.pid >20
  df1[df1.pid >20]
  print(type(df1[df1.pid >20]))
   df1.col3[df1.pid ==20] = 1
print()
