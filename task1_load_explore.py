import pandas as pd

df = pd.read_csv('churnguard_data.csv')

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df['Churn'].value_counts())
print(df['Contract'].unique())