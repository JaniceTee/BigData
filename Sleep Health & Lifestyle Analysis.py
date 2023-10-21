# IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

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

# Pie Chart
max_cat = 11
cat_plot = ['Gender', 'Occupation', 'BMI Category']
fig, axs = plt.subplots(nrows=1, ncols=3)
for i, var in enumerate(cat_plot):
    if i < len(axs.flat):
        cat_counts = df[var].value_counts()
        axs.flat[i].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
        axs.flat[i].set_title(f'{var} Distribition')
fig.tight_layout()
plt.show()

# Encode Labels of Object Datatypes
for col in df.select_dtypes(include=['object']).columns:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df[col].unique())
    df[col] = label_encoder.transform(df[col])
    print(f'{col}: {df[col].unique()}')

# Correlation Matrix and Heatmap of Predictors
print('\nCorrelation Matrix: ', '\n', df.corr())
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.title('Correlation Heatmap')
plt.show()

# DEVELOP AND EVALUATE MODELS
x = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

# SPLIT THE DATA INTO THE TRAIN AND TEST SETS.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=0)

# Remove Outliers
selected_columns = ['Heart Rate']
z_scores = np.abs(stats.zscore(x_train[selected_columns]))
threshold = 3
outlier_indices = np.where(z_scores > threshold)[0]
x_train = x_train.drop(x_train.index[outlier_indices])
y_train = y_train.drop(y_train.index[outlier_indices])

# 1.LOGISTIC REGRESSION MODEL.
lr = LogisticRegression(solver='liblinear', max_iter=400).fit(x_train, y_train)
lr_prediction = lr.predict(x_test)
lr_sq = lr.score(x, y)
print('\nLOGISTIC REGRESSION MODEL: ')
print(f'Coefficient of Determination: {lr_sq}')
print('Accuracy score: ', accuracy_score(y_test, lr_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, lr_prediction))
print('\nFrom Test Data: ')
for index in range(len(lr_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', lr_prediction[index])

# EVALUATE THE LOGISTIC REGRESSION MODEL
print('\n', 'Classification Report for Logistic Regression Model: ', '\n', classification_report(y_test, lr_prediction))

# 2.K-NEAREST NEIGHBORS (KNN) MODEL
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
knn_prediction = knn.predict(x_test)
knn_sq = knn.score(x, y)
print('KNN MODEL:')
print(f'Coefficient of Determination: {knn_sq}')
print('Accuracy score: ', accuracy_score(y_test, knn_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, knn_prediction))
print('\nFrom Test Data: ')
for index in range(len(knn_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', knn_prediction[index])

# EVALUATE THE KNN MODEL
print('\n', 'Classification Report for KNN Model: ', '\n', classification_report(y_test, knn_prediction))

# 3.NAIVE BAYES MODEL
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_prediction = nb.predict(x_test)
nb_sq = nb.score(x, y)
print('\nNAIVE BAYES MODEL:')
print(f'Coefficient of Determination: {nb_sq}')
print('Accuracy score: ', accuracy_score(y_test, nb_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, nb_prediction))
print('\nFrom Test Data: ')
for index in range(len(nb_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', nb_prediction[index])

# EVALUATE THE NAIVE BAYES MODEL
print('\n', 'Classification Report for Naive Bayes Model: ', '\n', classification_report(y_test, nb_prediction))

# 4.DECISION TREE CLASSIFIER MODEL
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_prediction = dtc.predict(x_test)
dtc_sq = dtc.score(x, y)
print('\nDECISION TREE CLASSIFIER MODEL:')
print(f'Coefficient of Determination: {dtc_sq}')
print('Accuracy score: ', accuracy_score(y_test, dtc_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, dtc_prediction))
print('\nFrom Test Data: ')
for index in range(len(dtc_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', dtc_prediction[index])

# EVALUATE THE DECISION TREE MODEL
print('\n', 'Classification Report for Decision Tree Model : ', '\n', classification_report(y_test, dtc_prediction))

# 5.RANDOM FOREST MODEL
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
rf_sq = rf.score(x, y)
print('\nRANDOM FOREST REGRESSION MODEL:')
print(f'Coefficient of Determination: {rf_sq}')
print('Accuracy score: ', accuracy_score(y_test, rf_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, rf_prediction))
print('\nFrom Test Data: ')
for index in range(len(rf_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', rf_prediction[index])

# EVALUATE THE RANDOM FOREST MODEL
print('\n', 'Classification Report for Random Forest Model: ', '\n', classification_report(y_test, rf_prediction))

# COMPARING MODELS
models = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Decision Tree', 'Random Forest']
acc_score = [accuracy_score(y_test, lr_prediction), accuracy_score(y_test, knn_prediction), accuracy_score(y_test, nb_prediction),
             accuracy_score(y_test, dtc_prediction), accuracy_score(y_test, rf_prediction)]
r_square = [lr_sq, knn_sq, nb_sq, dtc_sq, rf_sq]
x_axis = np.arange(len(models))
# Multi bar Chart
plt.bar(x_axis - 0.2, acc_score, width=0.4, label='Accuracy Score')
plt.bar(x_axis + 0.2, r_square, width=0.4, label='R Square Values')
plt.xticks(x_axis, models)
# Add legend
plt.legend()
# Display
plt.show()
