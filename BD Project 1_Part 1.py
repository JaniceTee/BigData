# INSTALL AND IMPORT ALL NECESSARY LIBRARIES.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# DATA PREPARATION/WRANGLING
df = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\BGDA 511\Heart_Disease_Prediction.csv")

print('Data Shape: ', df.shape)
print('\nData Types: ', '\n', df.dtypes)
print('\n', df.head())
print('\nCount of Null Values: ', '\n', df.isnull().sum())
print('\nChecking for Duplicate Records: ', '\n', df.duplicated())

# DATA EXPLORATION (D.E)
# D.E: 1.Visualization of Target Variable (Heart Disease (HD))
predicted_condition = df['HD'].value_counts()
print('\nPredicted Condition:', '\n', predicted_condition)
sns.countplot(x='HD', data=df, palette='hls')
plt.title('Heart Disease: Present(1)/Absent(0)')
plt.show()

# D.E: 2.Ratio of People Having and Not Having Heart Disease (PRESENT/ABSENT)
count_absence = len(df[df['HD'] == 0])
count_presence = len(df[df['HD'] == 1])
pct_of_absence = count_absence/(count_absence+count_presence)
print('\npercentage of patients not having heart disease is', pct_of_absence*100)
pct_of_presence = count_presence/(count_absence+count_presence)
print('percentage of of patients having heart disease is', pct_of_presence*100)

# D.E: 3.Description of Predictors
predictors = df[['Age', 'Sex', 'CP', 'BP', 'Cholesterol', 'FBS', 'EKG', 'HR', 'Angina', 'STD', 'SST', 'Vfluro',
                 'Thallium']]
print('\nDescription of Predictors: ', '\n', predictors.describe())

# D.E: 4.Correlation Matrix and Heatmap of Predictors
print('\nCorrelation Matrix: ', '\n', predictors.corr())
sns.heatmap(predictors.corr(), cmap="YlGnBu", annot=True)
plt.title('Predictors Correlation Heatmap')

# DEFINE X AND Y VARIABLES
x = df[['Age', 'Sex', 'CP', 'BP', 'Cholesterol', 'FBS', 'EKG', 'HR', 'Angina', 'STD', 'SST', 'Vfluro',
        'Thallium']].values
y = df['HD']

# SPLIT THE DATA INTO THE TRAIN AND TEST SETS.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=0)

# LOGISTIC REGRESSION MODEL.
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

#
# LOGISTIC REGRESSION ROC CURVE
logit_roc_auc = roc_auc_score(y_test, lr.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:, 1])
print("Area under ROC curve = {:.2f}".format(logit_roc_auc))
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()




