# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and read the csv file.
2. Import Decision tree classifier.
3. Fit the data in the model.
4. Find the accuracy score.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SRI SAI PRIYA. S
RegisterNumber: 212222240103
```
```
import pandas as pd
df=pd.read_csv("CSVs/Employee.csv")
df.head()
df.info()
df.isnull().sum()
df['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df['salary'])
df.head()
x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours',
      'time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=df['left']
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(Xtrain,Ytrain)
Ypred=dt.predict(Xtest)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(Ytest,Ypred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
# df.head()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475702/fd0d5932-db99-4661-8351-783f16f7b2b6)

# df.info()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475702/0d6a665c-2a72-4a01-8942-ffb7f51ecc88)

# df.isnull().sum()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475702/1ed9fd0e-74f5-4f12-9090-aecfec1691a5)

# df['left'].value_counts()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475702/c516a753-3974-4eed-83ef-4037b2da2c18)

# df["salary"]

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475702/9a3a576c-9744-4e2a-9920-d30c5668a869)

# x.head()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475702/524530c7-51da-42d5-8ad9-84129a408763)

# accuracy

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475702/f58bba59-1eec-448b-9353-2645be87c63f)

# dt.predict

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475702/086b157b-4b1c-41f8-a704-17067753f9c6)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
