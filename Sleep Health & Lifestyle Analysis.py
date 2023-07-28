# IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# DATA PREPARATION/WRANGLING
df = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\Sleep Health Analysis.csv")

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

# Check for Unique Values for Categorical Variables
print('\n', df['Gender'].unique())
print('\n', df['BMI Category'].unique())
print('\n', df['Occupation'].unique())
# Replace 'Normal Weight' with 'Normal'
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
print('\n', df['BMI Category'].unique())

# DATA EXPLORATION
# 1.Visualization of Target Variable (Sleep Disorder)
sleep_disorder = df['Sleep Disorder'].value_counts()
print('\nSleep Disorder:', '\n', sleep_disorder)
sns.countplot(x='Sleep Disorder', data=df, palette='cubehelix')
plt.title('Sleep Disorder: No Disorder, Sleep Apnea, Insomnia')
plt.show()

# 2.Description of Predictors
predictors = df.drop('Sleep Disorder', axis=1)
print('\nDescription of Predictors: ', '\n', predictors.describe())

# Plot Categorical Variables
# Bar Plot
cat_plot = ['Gender', 'Occupation', 'BMI Category']
fig, axs = plt.subplots(nrows=1, ncols=3)
axs.flatten()
for i, var in enumerate(cat_plot):
    sns.countplot(x=var, hue='Sleep Disorder', data=df, ax=axs[i])
    sns.histplot(x=var, hue='Sleep Disorder', data=df, ax=axs[i], multiple='fill', kde=False, element='bars')
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)
fig.tight_layout()
plt.show()

# Histogram
warnings.filterwarnings('ignore')
cat_plot = ['Gender', 'Occupation', 'BMI Category']
fig, axs = plt.subplots(nrows=1, ncols=3)
axs.flatten()
for i, var in enumerate(cat_plot):
    sns.histplot(x=var, hue='Sleep Disorder', data=df, ax=axs[i], multiple='fill', kde=False, element='bars')
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)
fig.tight_layout()
plt.show()