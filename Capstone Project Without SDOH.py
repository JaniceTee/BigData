import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# DATA CLEANING
df = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\Capstone\Capstone Research .csv")

# Renaming Features for Easy Analysis
# 1. Renaming Features
renamed_features = ['self_emp', 'no_of_emp_in_comp', 'tech_comp', 'emp_mh_benefits', 'emp_mh_care', 'emp_mh_discussion',
                    'emp_mh_resources', 'anonymity_protection', 'mh_leave_request_ease', 'mh_discussion_neg_impact',
                    'ph_discussion_neg_impact', 'mh_discussion_cowork',	'mh_discussion_supervisor',	'mh_eq_ph_employer',
                    'mh_conseq_coworkers', 'mh_coverage_self', 'mh_resources_self', 'mh_diagnosed_reveal_clients',
                    'mh_diagnosed_reveal_clients_negimpact', 'mh_diagnosed_reveal_cowork', 'mh_diagnosed_reveal_cowork_negimpact',
                    'mh_prod_impact', 'mh_prod_impact_perc', 'prev_emp', 'prev_emp_mh_benefits', 'prev_emp_mh_care',
                    'prev_emp_mh_discussion', 'prev_emp_mh_resources', 'prev_anonymity_protection',	'prev_mh_discussion_neg_impact',
                    'prev_ph_discussion_neg_impact', 'prev_mh_discussion_cowork', 'prev_mh_discussion_supervisor', 'prev_mh_eq_ph_employer',
                    'prev_mh_conseq_coworkers',	'future_ph_specification', 'why_whynot', 'future_mh_specification',	'why_whynot_two',
                    'mh_hurt_on_career', 'mh_neg_view_cowork', 'mh_sharing_friends_fam', 'mh_bad_response_workplace',
                    'mh_for_others_bad_response_workplace', 'family_hist', 'past_mh_hist', 'current_mh_disorder', 'diagnosed_conditions',
                    'possible_conditions', 'mh_diagnoses_professional',	'medically_diagnosed_conditions', 'professional_treatment_sought',
                    'mh_eff_treat_impact_on_work', 'mh_not_eff_treat_impact_on_work', 'age', 'sex', 'live_us_territory',
                    'work_us_territory', 'remote_work']
df.columns = renamed_features
print(df.iloc[:15, :10])

# Dropping Features With Over 50% Missing Values
df = df.drop(columns=['mh_coverage_self', 'mh_resources_self', 'mh_diagnosed_reveal_clients', 'mh_diagnosed_reveal_clients_negimpact',
                      'mh_diagnosed_reveal_cowork', 'mh_diagnosed_reveal_cowork_negimpact', 'mh_prod_impact', 'mh_prod_impact_perc'])

# Dropping Possible Confounding Variables
df = df.drop(columns=['why_whynot', 'why_whynot_two', 'diagnosed_conditions', 'possible_conditions'])

# Eliminating Missing Values and Checking For the Number of Unique Values
missing_values = df.isna()
df.fillna(value=pd.NA, inplace=True)
placeholder_value = "unknown"
object_columns = df.select_dtypes(include=['object']).columns
df[object_columns] = df[object_columns].fillna(placeholder_value)
float_columns = df.select_dtypes(include=['float']).columns
df[float_columns] = df[float_columns].fillna(placeholder_value)
frame = pd.concat([df.isnull().sum(), df.nunique(), df.dtypes], axis=1, sort=False)
print(frame)

# Remove Age Outlier
mean_age = df[(df['age'] >= 18) | (df['age'] <= 75)]['age'].mean()
print("Mean Age is :", mean_age)
df['age'].replace(to_replace=df[(df['age'] < 18) | (df['age'] > 75)]['age'].to_list(), value=mean_age, inplace=True)
print(df.describe())

# The sex variable has many responses so they will be encoded for visualization.
# Male - 0, Female - 1, Other - 2, Unknown - 3
df['sex'].replace(to_replace=['Male', 'male', 'Male ', 'M', 'm', 'man', 'male 9:1 female, roughly','Male (cis)',
                              'Cis male', 'Male.', 'Man', 'Sex is male', 'cis male', 'Malr', 'Dude',
                              "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would "
                              "of answered yes please. Seriously how much text can this take? ",
                              'mail', 'M|', 'Male/genderqueer', 'male ', 'Cis Male', 'Male (trans, FtM)',
                              'cisdude', 'cis man', 'MALE'], value=0, inplace=True)
df['sex'].replace(to_replace=['Female', 'female', 'I identify as female.', 'female ',
                              'Cis female ', 'Transitioned, M2F', 'Genderfluid (born female)',
                              'Female or Multi-Gender Femme', 'Female ', 'woman', 'female/woman',
                              'Cisgender Female', 'fem', 'Female (props for making this a freeform field, though)',
                              ' Female', 'Cis-woman', 'female-bodied; no feelings about gender', 'AFAB', 'F', 'f',
                              'Woman', 'fm', 'Female assigned at birth '], value=1, inplace=True)
df['sex'].replace(to_replace=['Bigender', 'non-binary', 'Other/Transfeminine',  'Androgynous',
                              'Other', 'nb masculine', 'none of your business', 'genderqueer', 'Human', 'Genderfluid',
                              'Enby', 'genderqueer woman', 'mtf', 'Queer','Agender', 'Fluid', 'Nonbinary', 'human',
                              'Unicorn', 'Genderqueer', 'Genderflux demi-girl', 'Transgender woman'], value=2, inplace=True)
df['sex'].replace(to_replace=['unknown'], value=3, inplace=True)

# Tech Company contains 'unknown' so it has to be encoded
# 0 - No, 1 - Yes, 2 - Unknown
df['tech_comp'].replace(to_replace=['unknown'], value=2.0, inplace=True)
df['tech_comp'].replace(to_replace=['1.0'], value=1, inplace=True)
df['tech_comp'].replace(to_replace=['0.0'], value=0, inplace=True)

# Number of Employees has incoherent data so it must be encoded for visualization
df['no_of_emp_in_comp'].replace(to_replace=['5-Jan'], value='1-25', inplace=True)
df['no_of_emp_in_comp'].replace(to_replace=['25-Jun'], value='6-25', inplace=True)
df['no_of_emp_in_comp'].replace(to_replace=['100-500'], value='101-500', inplace=True)
df['no_of_emp_in_comp'].replace(to_replace=['500-1000'], value='501-1000', inplace=True)
df['no_of_emp_in_comp'].replace(to_replace=['More than 1000'], value='>1000', inplace=True)

# DATA ENCODING
# Encode input variables pre-selected with the SDOH framework
# Combine related values of variables for accurate processing
df['emp_mh_benefits'].replace(to_replace=['Not eligible for coverage / N/A'], value='No', inplace=True)
df['mh_bad_response_workplace'].replace(to_replace=['Yes, I experienced', 'Yes, I observed'], value='Yes', inplace=True)

# Create a dictionary to store the mappings
label_mappings = {}

# List of columns to be encoded
columns_to_encode = ['no_of_emp_in_comp', 'emp_mh_benefits', 'emp_mh_care', 'emp_mh_discussion',
                     'emp_mh_resources', 'anonymity_protection', 'mh_leave_request_ease', 'mh_discussion_neg_impact',
                     'ph_discussion_neg_impact', 'mh_discussion_cowork',	'mh_discussion_supervisor',	'mh_eq_ph_employer',
                     'mh_conseq_coworkers', 'prev_emp_mh_benefits', 'prev_emp_mh_care', 'prev_emp_mh_discussion',
                     'prev_emp_mh_resources', 'prev_anonymity_protection',	'prev_mh_discussion_neg_impact',
                     'prev_ph_discussion_neg_impact', 'prev_mh_discussion_cowork', 'prev_mh_discussion_supervisor',
                     'prev_mh_eq_ph_employer', 'prev_mh_conseq_coworkers', 'future_ph_specification', 'future_mh_specification',
                     'mh_hurt_on_career', 'mh_neg_view_cowork', 'mh_sharing_friends_fam', 'mh_bad_response_workplace',
                     'mh_for_others_bad_response_workplace', 'family_hist', 'past_mh_hist', 'current_mh_disorder',
                     'mh_diagnoses_professional', 'mh_eff_treat_impact_on_work', 'mh_not_eff_treat_impact_on_work',
                     'live_us_territory', 'work_us_territory', 'remote_work']
for col in columns_to_encode:
    label_encoder = preprocessing.LabelEncoder()
    unique_values = df[col].unique()
    label_encoder.fit(unique_values)
    df[col] = label_encoder.transform(df[col])
    # Store the mappings in the dictionary
    label_mappings[col] = {original: encoded for original, encoded in zip(unique_values, label_encoder.transform(unique_values))}
    print(f'{col}: {df[col].unique()}')
# Print the label mappings
print("\nLabel Mappings:")
for col, mapping in label_mappings.items():
    print(f'{col}: {mapping}')

# CORRELATION MATRIX FOR FEATURE SELECTION (Correlation between Independent variables and Independent and Dependent Vr.)
variables = df[['no_of_emp_in_comp', 'emp_mh_benefits', 'emp_mh_care', 'emp_mh_discussion',
               'emp_mh_resources', 'anonymity_protection', 'mh_leave_request_ease', 'mh_discussion_neg_impact',
               'ph_discussion_neg_impact', 'mh_discussion_cowork',	'mh_discussion_supervisor',	'mh_eq_ph_employer',
               'mh_conseq_coworkers', 'prev_emp_mh_benefits', 'prev_emp_mh_care', 'prev_emp_mh_discussion',
               'prev_emp_mh_resources', 'prev_anonymity_protection',	'prev_mh_discussion_neg_impact',
               'prev_ph_discussion_neg_impact', 'prev_mh_discussion_cowork', 'prev_mh_discussion_supervisor',
               'prev_mh_eq_ph_employer', 'prev_mh_conseq_coworkers', 'future_ph_specification', 'future_mh_specification',
               'mh_hurt_on_career', 'mh_neg_view_cowork', 'mh_sharing_friends_fam', 'mh_bad_response_workplace',
               'mh_for_others_bad_response_workplace', 'family_hist', 'past_mh_hist', 'current_mh_disorder',
               'mh_eff_treat_impact_on_work', 'mh_not_eff_treat_impact_on_work', 'live_us_territory', 'work_us_territory',
               'age', 'sex', 'self_emp', 'tech_comp', 'remote_work', 'professional_treatment_sought', 'prev_emp',
               'mh_diagnoses_professional']]
print('\nCorrelation Matrix: ', '\n', variables.corr())
sns.heatmap(variables.corr(), cmap="YlGnBu", annot=True)
plt.title('Correlation Heatmap')
plt.show()
# Feature Selection
correlation_matrix = variables.corr()
# Set your correlation threshold
correlation_threshold = 0.05
# Find the features highly correlated with the dependent variable
target_correlations = correlation_matrix['mh_diagnoses_professional']
significant_features = target_correlations[abs(target_correlations) > correlation_threshold]
# Print the selected features
print('\nSignificant Features: ', '\n', significant_features)

# DEVELOPMENT OF MODELS
x = df[['emp_mh_benefits', 'emp_mh_care', 'emp_mh_discussion', 'emp_mh_resources', 'mh_eq_ph_employer', 'mh_conseq_coworkers',
        'prev_emp_mh_benefits', 'prev_mh_discussion_neg_impact', 'prev_mh_discussion_supervisor', 'prev_mh_conseq_coworkers',
        'future_mh_specification', 'mh_hurt_on_career', 'mh_neg_view_cowork', 'mh_sharing_friends_fam', 'mh_for_others_bad_response_workplace',
        'family_hist', 'past_mh_hist', 'current_mh_disorder', 'mh_eff_treat_impact_on_work', 'mh_not_eff_treat_impact_on_work',
        'sex', 'professional_treatment_sought']].values
y = df['mh_diagnoses_professional']
# Split the Data into the Training and Testing Sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=0)

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
knn = KNeighborsClassifier(n_neighbors=4)
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

# LOGISTIC REGRESSION ROC CURVE
logit_roc_auc = roc_auc_score(y_test, lr.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:, 1])
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

# KNN ROC CURVE
logit_roc_auc = roc_auc_score(y_test, knn.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='K-Nearest Neighbor AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for K-Nearest Neighbor')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# NAIVE BAYES ROC CURVE
logit_roc_auc = roc_auc_score(y_test, nb.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, nb.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Naive Bayes AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# DECISION TREE ROC CURVE
logit_roc_auc = roc_auc_score(y_test, dtc.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, dtc.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for Decision Tree Classifier')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# RANDOM FOREST ROC CURVE
logit_roc_auc = roc_auc_score(y_test, rf.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for Random Forest Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# SHAP values to determine feature importance to Decision Tree Model
explainer = shap.TreeExplainer(dtc)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test)
plt.show()

# SHAP values to determine feature importance to Random Forest Model
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test)
plt.show()
