import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model


df = pd.read_csv('churn.csv')  #---- reading from csv file 
print(df)#-- printing the rows and columns 

churn= df.groupby('Churn').mean()[['Age','Tenure']]#--- grouping by chrun = 1 or 0 and printing 
#the avg of age and the time they were  with the company
print(churn)

X = df[['Age', 'Tenure']]  # Features
y = df['Churn']            # Target
#applying train text split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
# Output the results
print("Training Set Features:\n", X_train)
print("\nTraining Set Target:\n", y_train)
print("\nTesting Set Features:\n", X_test)
print("\nTesting Set Target:\n", y_test)
#choosing the model 
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
prediction_test= reg.predict(X_test)#pridicting the values for who churned and texting the accurary
print(y_test,prediction_test)

