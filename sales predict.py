

`Load the dataset
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

"""Uploading data"""

df = pd.read_csv('/content/advertising.csv')

df = pd.DataFrame()

df.head(5)

df.tail(5)

df.shape

df.isnull()

df.isnull().sum()

df.columns

df.info()

df.size

df.describe()

df.duplicated()

df.duplicated().sum()

if isinstance(df, pd.DataFrame):
    x = df.iloc[:, :-1]
else:
    print("Error: 'df' is not a Pandas DataFrame. Check its assignment.")

x=df.iloc[:,:-1]

x

df.empty

df.shape

df = pd.read_csv('/content/advertising.csv')
x = df.iloc[:, :-1]
y=df.iloc[:,-1]

x

y

x = df.iloc[:,0]

x

y=df.iloc[:,0]

y

df.head(10)

df.tail(10)

df.corr()

import seaborn as sns
sns.pairplot(df)
plt.show()

df = pd.read_csv('/content/advertising.csv')
sns.jointplot(x=df.columns[1], y=df.columns[2], data=df, kind='reg')
plt.show()

sns.pairplot(df, hue='Sales')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Split data into training and testing sets
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train

y_train

X_test

y_test

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

sns.scatterplot(x=y_test, y=y_pred, hue=y_test > 1500)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()
