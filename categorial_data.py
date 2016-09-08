import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                  ['red', 'L', 13.5, 'class2'],
                  ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classLabel']
print(df)

# mapping ordinal features
size_mapping = {'XL': 3, 'L': 2,  'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classLabel']))}
print(class_mapping)

df['classLabel'] = df['classLabel'].map(class_mapping)
print(df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classLabel'] = df['classLabel'].map(inv_class_mapping)
print(df)

class_encoder = LabelEncoder()
y = class_encoder.fit_transform(df['classLabel'].values)
print(y)
print(class_encoder.inverse_transform(y))

x = df [['color', 'size', 'price']].values
class_encoder = LabelEncoder()
x[:, 0] = class_encoder.fit_transform(x[:, 0])
print(x)

# one-hot encoding
one_encoder = OneHotEncoder(categorical_features=[0])
print(one_encoder.fit_transform(x).toarray())
# this one-hot is more readable
print(pd.get_dummies(df[['price', 'color', 'size']]))