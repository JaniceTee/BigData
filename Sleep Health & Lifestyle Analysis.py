# IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# DATA PREPARATION/WRANGLING
df = pd.read_csv(r"C:\Users\princ\Downloads\Sleep_health_and_lifestyle_dataset.csv")

print('Data Shape: ', df.shape)
print('\nData Types: ', '\n', df.dtypes)
print('\n', df.head())
print('\nCount of Null Values: ', '\n', df.isnull().sum())
print('\nChecking for Duplicate Records: ', '\n', df.duplicated())

# Drop identifier 'Person ID' column
df.drop(columns='Person ID', inplace=True)
print('\n', df.head())

# Split Blood Pressure Column into 2 and Drop Blood Pressure column
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df.drop(columns='Blood Pressure', inplace=True)
print('\n', df.head())

# Check for Unique Values for Arrays
print('\n', df['BMI Category'].unique())
# Replace 'Normal Weight' with 'Normal'
