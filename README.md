# Airlines Delay Prediction Project

## Overview
This project focuses on analyzing and predicting flight delays, aiming to enhance our understanding of the factors influencing delays and create a predictive model using machine learning techniques. The primary objective is to implement statistical techniques, feature selection, and a machine learning model to accurately predict flight delays.

## Data Dictionary
- **Flight**        : Flight ID 
- **Time**          : Time of Departure (In Mins continuously from Midnight to Next Midnight)
- **Length**        : Length of the Flight (In Mins)
- **Airline**       : Airline Unique Code
- **AirportFrom**   : Which Airport the flight flew from
- **AirportTo**     : Which Airport the flight flew to
- **DayOfWeek**     : Day of the Week of the flight (Starting Sunday
- **Class Delayed** : (1) or Not (0)

## Project Highlights

### Exploratory Data Analysis (EDA)
- Utilized statistical techniques such as boxplot and histplot to gain insights into data distribution and identify potential patterns.
- Investigated the correlation between different features to understand their relationships and potential impact on flight delays.
  
#### Importing Necessary Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
#### Loading the Data
```
df = pd.read_excel('airlines_delay.xlsx', sheet_name='data')
```
#### Having a First Look at the Data
![3](https://github.com/himehul/Airlines-Delay-Prediction-Project/assets/139626006/3b34da98-18a9-4ebe-a055-a92fafc01b8d)
#
![5](https://github.com/himehul/Airlines-Delay-Prediction-Project/assets/139626006/4e325575-f460-44d0-ab1a-77bca14aa556)

#
#### Checking the Correlation
![4](https://github.com/himehul/Airlines-Delay-Prediction-Project/assets/139626006/5dcfe38a-97eb-4d32-8f50-a22fbe6b6ea4)
#
As we can see we don't have any strong correlation coefficient.
#
# EDA examples
```
sns.boxplot(data=df, x='Class', y='Time')
```
#
![1](https://github.com/himehul/Airlines-Delay-Prediction-Project/assets/139626006/c1f911f3-925a-435e-9bb0-094933717eba)

#
```
sns.histplot(data=df, x='Time', bins=100, hue='Class')
```
#
![2](https://github.com/himehul/Airlines-Delay-Prediction-Project/assets/139626006/1903154c-c605-4134-974e-e5d057d73aae)

#
### Feature Engineering
- Applied the correlation filter method for feature selection, resulting in a significant 5% improvement in accuracy.
- Implemented mean encoding to convert categorical data (such as AirportFrom and AirportTo) into numerical format for efficient use in machine learning models.

```python
mean_from_encoding = df.groupby('AirportFrom')['Class'].mean()
df['AirportFrom_Encoded'] = df['AirportFrom'].map(mean_from_encoding)

mean_to_encoding = df.groupby('AirportTo')['Class'].mean()
df['AirportTo_Encoded'] = df['AirportTo'].map(mean_to_encoding)
```

### Machine Learning Model
- Utilized the random forest algorithm as the primary machine learning model for predicting flight delays.
- Employed one-hot encoding for the 'Airline' feature and dropped unnecessary columns for model training.

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# One-hot encoding for 'Airline'
airline_dummies = pd.get_dummies(df['Airline'], drop_first=True)
df = pd.concat([df, airline_dummies], axis=1)
df.drop(['Airline', 'AirportFrom', 'AirportTo'], axis=1, inplace=True)

# Model training
X = df[['Time', 'AirportFrom_Encoded', 'AirportTo_Encoded', 'WN']]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
```

### Model Evaluation
- Assessed model performance using confusion matrix, classification report, and ROC-AUC score.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Model evaluation
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))
print(roc_auc_score(y_test, rfc_pred))
```

## Conclusion
This project provides a comprehensive analysis of flight delays, including data exploration, feature engineering, and the implementation of a random forest classifier for prediction. The insights gained from this project can be valuable for stakeholders in the aviation industry and contribute to the development of proactive measures to mitigate flight delays.
