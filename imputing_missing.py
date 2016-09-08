import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import datasets


df = pd.read_csv('data.csv')
print(df)
print(df.isnull().sum())
print(df.dropna())
print(df.dropna(axis=1))

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)