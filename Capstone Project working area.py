# Eliminating Missing Values
missing_values = df.isna()
df.fillna(value=pd.NA, inplace=True)
df.to_csv("modified_dataset.csv", index=False)
print('\nCount of Null Values: ', '\n', df.isnull().sum())
print(df.iloc[:15, :10])

# Dropping Comments & Work and Live in U.S varibales
df.drop(columns='Why or why not?', inplace=True)
df.drop(columns='What country do you live in?', inplace=True)
df.drop(columns='What country do you work in?', inplace=True)

# Renaming Features for Easy Analysis
# 1. Renaming Features
renamed_features = ['self_empl_flag', 'comp_no_empl',  'tech_comp_flag', 'tech_role_flag', 'mh_coverage_flag',
                    'mh_coverage_awareness_flag', 'mh_employer_discussion', 'mh_resources_provided', 'mh_anonimity_flag',
                    'mh_medical_leave', 'mh_discussion_neg_impact', 'ph_discussion_neg_impact', 'mh_discussion_cowork',
                    'mh_discussion_supervisor', 'mh_eq_ph_employer', 'mh_conseq_coworkers', 'mh_coverage_flag2',
                    'mh_online_res_flag', 'mh_diagnosed&reveal_clients_flag', 'mh_diagnosed&reveal_clients_impact',
                    'mh_diagnosed&reveal_cowork_flag', 'mh_cowork_reveal_neg_impact', 'mh_prod_impact', 'mh_prod_impact_perc',
                    'prev_employers_flag', 'prev_mh_benefits', 'prev_mh_benefits_awareness', 'prev_mh_discussion',
                    'prev_mh_resources', 'prev_mh_anonimity', 'prev_mh_discuss_neg_conseq', 'prev_ph_discuss_neg_conseq',
                    'prev_mh_discussion_cowork', 'prev_mh_discussion_supervisor', 'prev_mh_importance_employer',
                    'prev_mh_conseq_coworkers', 'future_ph_specification', 'future_mh_specification',
                    'why/why_not2', 'mh_hurt_on_career', 'mh_neg_view_cowork', 'mh_sharing_friends/fam_flag',
                    'mh_bad_response_workplace', 'mh_for_others_bad_response_workplace', 'mh_family_hist', 'mh_disorder_past',
                    'mh_disorder_current', 'yes:what_diagnosis?', 'maybe:whats_your_diag', 'mh_diagnos_proffesional',
                    'yes:condition_diagnosed', 'mh_sought_proffes_treatm',  'mh_eff_treat_impact_on_work',
                    'mh_not_eff_treat_impact_on_work', 'age', 'sex', 'live_us_territory', 'work_us_territory', 'work_position','remote_flag']

df.columns = renamed_features
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.iloc[:15, :10])

# DATA ENCODING
# Check for Unique Values for  Variables
unique_values = {}
for column in df.columns:
    unique_values[column] = df[column].unique()

# Print unique values for each column
for column, values in unique_values.items():
    print(f"Unique values in '{column}':")
    for value in values:
        print(value)
    print()

# Renaming Features for Easy Analysis
# 1. Renaming Features
renamed_features = ['self_emp', 'no_of_emp_in_comp', 'tech_comp', 'emp_mh_benefits', 'emp_mh_care', 'emp_mh_discussion',
                    'emp_mh_resources', 'anonymity_protection', 'mh_leave_request_ease',	'mh_discussion_neg_impact',
                    'ph_discussion_neg_impact', 'mh_discussion_cowork',	'mh_discussion_supervisor',	'mh_eq_ph_employer',
                    'mh_conseq_coworkers', 'mh_coverage_self', 'mh_resources_self', 'mh_diagnosed_reveal_clients',
                    'mh_diagnosed_reveal_clients_negimpact', 'mh_diagnosed_reveal_cowork', 'mh_diagnosed_reveal_cowork_negimpact',
                    'mh_prod_impact', 'mh_prod_impact_perc', 'prev_emp', 'prev_emp_mh_benefits',	'prev_emp_mh_care',
                    'prev_emp_mh_discussion', 'prev_emp_mh_resources', 'prev_anonymity_protection',	'prev_mh_discussion_neg_impact',
                    'prev_ph_discussion_neg_impact',	'prev_mh_discussion_cowork', 'prev_mh_discussion_supervisor', 'prev_mh_eq_ph_employer',
                    'prev_mh_conseq_coworkers',	'future_ph_specification', 'why_whynot', 'future_mh_specification',	'why_whynot_two',
                    'mh_hurt_on_career',	'mh_neg_view_cowork', 'mh_sharing_friends_fam',	'mh_bad_response_workplace',
                    'mh_for_others_bad_response_workplace', 'family_hist' 'past_mh_hist','current_mh_disorder', 'diagnosed_conditions',
                    'possible_conditions', 'mh_diagnoses_professional',	'medically_diagnosed_conditions', 'professional_treatment_sought',
                    'mh_eff_treat_impact_on_work', 'mh_not_eff_treat_impact_on_work', 'age',	'sex', 'live_us_territory',
                    'work_us_territory', 'remote_work']
df.columns = renamed_features
print(df.iloc[:15, :10])

# Dropping Features With Over 50% Missing Values
df.drop(columns='Do you know local or online resources to seek help for a mental health disorder?', inplace=True)
df.drop(columns='If you have been diagnosed or treated for a mental health disorder, '
                'do you ever reveal this to clients or business contacts?', inplace=True)
df.drop(columns='If you have revealed a mental health issue to a client or business contact, '
                'do you believe this has impacted you negatively?', inplace=True)
df.drop(columns='If you have been diagnosed or treated for a mental health disorder, '
                'do you ever reveal this to coworkers or employees?', inplace=True)
df.drop(columns='If you have revealed a mental health issue to a coworker or employee, '
                'do you believe this has impacted you negatively?', inplace=True)
df.drop(columns='Do you believe your productivity is ever affected by a mental health issue?', inplace=True)
df.drop(columns='If yes, what percentage of your work time (time performing primary '
                'or secondary job functions) is affected by a mental health issue?', inplace=True)

# Dropping Possible Confounding Variables
df.drop(columns='Why would you be willing or unwilling to bring up a physical '
                'health issue with a potential employer in an interview?', inplace=True)
df.drop(columns='Why would you or would you not bring up a mental health '
                'issue with a potential employer in an interview?', inplace=True)
df.drop(columns='If yes, what condition(s) have you been diagnosed with?', inplace=True)
df.drop(columns='If maybe, what condition(s) do you believe you have?', inplace=True)

# Histogram of Sex
df.sex.hist()
plt.title('Histogram of Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.show()

value_counts = df['live_us_territory'].value_counts()
print("Frequency of Unique Values in 'State' variable:")
print(value_counts)

value_counts = df['sex'].value_counts()
print("Frequency of Unique Values in 'sex' variable:")
print(value_counts)

value_counts = df['age'].value_counts()
print("Frequency of Unique Values in 'age' variable:")
print(value_counts)

# Visualize Mental Health Conditions diagnosed by medical professional
print(df['medically_diagnosed_conditions'].unique())
value_counts = df['medically_diagnosed_conditions'].value_counts()
value_counts.plot(kind='bar')
plt.xticks(rotation=90)
plt.margins(0.2)
plt.title('Bar Graph of Medically Diagnosed Mental Health Conditions Occurrence')
plt.xlabel('Mental health conditions')
plt.ylabel('Frequency')
plt.show()

x_train_df = pd.DataFrame(x_train)
imp_df = pd.DataFrame({
    'Feature Name': x_train.columns,
    'Importance': dtc.feature_importances_
})
fi = imp_df.sort_values(by='Importance', ascending=False)
fi2 = fi.head(10)
plt.figure(figsize=(10, 8))
sns.barplot(df=fi2, x='Importance', y='Feature Name')
plt.title('Top Feature Importance for Each Attribute (Decision Tree)', fontsize=18)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Feature Name', fontsize=16)
plt.show()

# 5.RANDOM FOREST MODEL WITH CROSS VALIDATION
rf = RandomForestClassifier(class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [100, 200],
    'min_samples_split': [None, 5, 4],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [0, 42]
}
grid_search = GridSearchCV(dtc, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
rf = RandomForestClassifier(random_state=0, max_features='sqrt', n_estimators=1000, max_depth=100, class_weight='balanced')
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

# 4.DECISION TREE CLASSIFIER MODEL WITH CROSS VALIDATION
dtc = DecisionTreeClassifier(class_weight='balanced')
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4],
    'random_state': [0, 42]
}
# Cross-validation to determine best hyperparameters
grid_search = GridSearchCV(dtc, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
dtc = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=3, min_samples_split=2, class_weight='balanced')
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