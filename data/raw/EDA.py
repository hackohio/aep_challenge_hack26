import pandas as pd

df = pd.read_csv("Ambulance_Servi_Pairwis_Clip_attributes.csv")

print(df.info())
print(df.head())
print(df.describe())
print(df.isnull().sum())