import pandas as pd
from sklearn.datasets import load_wine
wine=load_wine()

df=pd.DataFrame(wine.data, columns=wine.feature_names)

df["target"]=wine.target

print("--- First Five Rows ---")
print(df.head())

print("\n --- Dataset Shape ---")
print(df.shape)

print("\n --- Dataset Information ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Column Names ---")
print(df.columns)

print("\n --- Target Distribution ---")
print(df["target"].value_counts())