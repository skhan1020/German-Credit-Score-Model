# German Credit Score Model : Determining the most efficient model to predict whether customers
# of a particular bank are creditable or not. Training and test sets are in TrainingSet.csv and TestSet.csv respectively.
# Data is preprocessed to include the most relevant input features.
# Logistic Regression, Decision Tree, Random Forest and Gradient Boosted Classifiers are used for evaluation. 
# Area under the ROC curve is chosen as the metric to select the best model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel
sns.set_style('white')

# Load the training and test sets
a = list(np.arange(1,22))
credit_training = pd.read_csv('TrainingSet.csv', usecols = a)
credit_test = pd.read_csv('TestSet.csv', usecols = a)


### --- Exploratory Analysis and Data Preprocessing --- ###

# --- List of Variables ---- #
print(' Columns : ', credit_training.columns)
print('\n')

# Target values : Creditable - 1 (70 %), Non-Creditable - 0 (30 %)
print(credit_training['Creditability'].value_counts())
print('\n')

# Account Balance 
print(credit_training['Account.Balance'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Account.Balance'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Squrare Value, P-Value', chi2, p)

# Barplot of Account Balance categories 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Account.Balance'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Account Balance',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')

# Histogram of Duration of Credit - months
plt.figure(figsize = (10,6))
credit_training['Duration.of.Credit..month.'].plot.hist(color = 'k', bins = 50, alpha = 0.5)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Duration of Credit month', fontsize=20)
plt.ylabel('Number of Customers', fontsize=20)
print('\n')

# Payment Status of Previous Credit
print(credit_training['Payment.Status.of.Previous.Credit'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Payment.Status.of.Previous.Credit'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Payment Status of Previous Credit 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Payment.Status.of.Previous.Credit'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Payment Status of Previous Credit',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')

# Purpose categorical values : before mapping
print(credit_training['Purpose'].value_counts())

# Purpose categorical values : 1 & 2 mapped to 1, 3 to 2, 4 to 3
credit_training['Purpose'] = credit_training['Purpose'].map({1:1, 2:1, 3:2, 4:3})
print(credit_training['Purpose'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Purpose'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)


# Barplot of Purpose categories 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Purpose'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Purpose',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')

# Histogram of Credit Amount
plt.figure(figsize = (10,6))
credit_training['Credit.Amount'].plot.hist(color = 'k', bins = 50, alpha = 0.5)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Credit Amount', fontsize =20)
plt.ylabel('Number of Customers', fontsize =20)
print('\n')

# Value of Savings Stocks categorical values : before mapping; Too few obs. in 3 and 4
print(credit_training['Value.Savings.Stocks'].value_counts())

# Value of Savings Stocks categorical values : 2, 3, 4 mapped to 2
credit_training['Value.Savings.Stocks'] = credit_training['Value.Savings.Stocks'].map({1:1, 2:2, 3:2, 4:2})
print(credit_training['Value.Savings.Stocks'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Value.Savings.Stocks'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Savings Stocks Value categories 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Value.Savings.Stocks'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Value of Savings/Stocks',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Length of Current employment values ( yearly )
print(credit_training['Length.of.current.employment'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Length.of.current.employment'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)


# Barplot of Current employment values ( yearly ) 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Length.of.current.employment'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Length of Current Employment',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')

# Instalment per cent categorical values 
print(credit_training['Instalment.per.cent'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Instalment.per.cent'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of instalment per cent 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Instalment.per.cent'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Instalment per cent',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')

# Sex - Marital Status values
print(credit_training['Sex...Marital.Status'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Sex...Marital.Status'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Sex and Marital Status 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Sex...Marital.Status'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Sex and Marital Status',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Guarantors
print(credit_training['Guarantors'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Guarantors'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Guarantors 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Guarantors'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Guarantors',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Duration in Current Address values
print(credit_training['Duration.in.Current.address'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Duration.in.Current.address'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Duration in Address values 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Duration.in.Current.address'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Duration in Current Address (years)',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
plt.show()
print('\n')

# Concurrent credits categorical values
print(credit_training['Concurrent.Credits'].value_counts())


# Contingency Table 
tab = pd.crosstab(credit_training['Concurrent.Credits'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Concurrent credits 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Concurrent.Credits'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Concurrent Credits',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Most valuable available asset categorical values
print(credit_training['Most.valuable.available.asset'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Most.valuable.available.asset'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of asset 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Most.valuable.available.asset'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Most Valuable Available Asset',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')

# Histogram of Age of customers
plt.figure(figsize = (10,6))
credit_training['Age..years.'].plot.hist(color = 'k', bins = 50, alpha = 0.5)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Age', fontsize = 20)
plt.ylabel('Number of Customers', fontsize = 20)
print('\n')


# Apartment categorical values
print(credit_training['Type.of.apartment'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Type.of.apartment'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of type of apartment categories 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Type.of.apartment'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Type of Apartment',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Credits at the Bank (categorical values)
print(credit_training['No.of.Credits.at.this.Bank'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['No.of.Credits.at.this.Bank'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Total Credits 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['No.of.Credits.at.this.Bank'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Total No. of Credits at this Bank',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Occupation categories : Drop this feature since there is only one category
print(credit_training['Occupation'].value_counts())
credit_training = credit_training.drop('Occupation', axis = 1)
credit_test = credit_test.drop('Occupation', axis = 1)
print('\n')

# Dependent feature values
print(credit_training['No.of.dependents'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['No.of.dependents'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Dependents 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['No.of.dependents'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('No. of Dependents',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Telephone feature values 
print(credit_training['Telephone'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Telephone'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Telephone feature values 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Telephone'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Telephone',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Foreign Worker feature values 
print(credit_training['Foreign.Worker'].value_counts())

# Contingency Table 
tab = pd.crosstab(credit_training['Foreign.Worker'], credit_training['Creditability'])
print(tab)
chi2, p, dof, exp = chi2_contingency(tab)
print('Chi-Square Value, P-Value', chi2, p)

# Barplot of Foregin Worker feature values 
fig = plt.figure(figsize = (6,6))
stacked = tab.stack().reset_index().rename(columns={0:'value'})
sns.barplot(x = stacked['Foreign.Worker'], y = stacked.value, hue = stacked['Creditability'])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Foreign Worker',fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()
print('\n')


# Summary of statistics for continuous variables:
print(credit_training['Age..years.'].describe())
print(credit_training['Duration.of.Credit..month.'].describe())
print(credit_training['Credit.Amount'].describe())
print('\n')

# Plotting the Correlation Matrix : Ignore Duration of Credit Month henceforth

data = credit_training[['Account.Balance','Payment.Status.of.Previous.Credit','Purpose','Most.valuable.available.asset','Value.Savings.Stocks','Concurrent.Credits','Length.of.current.employment','Type.of.apartment','Age..years.','Credit.Amount','Duration.of.Credit..month.']]

fig, ax = plt.subplots(figsize = (10,8))
corr = data.corr()
sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values, mask = np.zeros_like(corr, dtype = np.bool), cmap = sns.diverging_palette(220, 10, as_cmap = True), square = True, ax = ax)
plt.gcf().subplots_adjust(bottom=0.35)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Scaling the data
scaler = MinMaxScaler()

X_train = credit_training[['Account.Balance','Payment.Status.of.Previous.Credit','Purpose','Most.valuable.available.asset','Value.Savings.Stocks','Concurrent.Credits','Length.of.current.employment','Type.of.apartment','Age..years.','Credit.Amount']]
y_train = credit_training.loc[:, 'Creditability']

X_test = credit_test[['Account.Balance','Payment.Status.of.Previous.Credit','Purpose','Most.valuable.available.asset','Value.Savings.Stocks','Concurrent.Credits','Length.of.current.employment','Type.of.apartment','Age..years.','Credit.Amount']]
y_test = credit_test.loc[:,'Creditability']

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


### ---- Dummy Classifier invoke to test for accuracy ---- ###

print('Dummy Classifier evaluation')
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train_scaled, y_train)
y_dummy_majorty = dummy_majority.predict(X_test_scaled)
print('Acccuracy of Dummy classifier on test set {:.3f}\n'.format(dummy_majority.score(X_test_scaled, y_test)))



### ---- Logistic Regression Classifier ---- ### 
for this_C in [0.1,1,100]:
    
    print('Regulatization parameter {:.2f}'.format(this_C))
    
    clf = LogisticRegression(C = this_C).fit(X_train_scaled, y_train)
    y_predict = clf.predict(X_test_scaled)
    y_score_clf = clf.decision_function(X_test_scaled)

    

    print('Acccuracy of Logistic Regression classifier on training set', clf.score(X_train_scaled, y_train))
    print('Accuracy of Logistic Regression classifier on test set', clf.score(X_test_scaled, y_test))
    print('Precision', precision_score(y_test, y_predict))
    print('Recall', recall_score(y_test, y_predict))
    print('F1-score', f1_score(y_test, y_predict))
    
    fpr_clf, tpr_clf, _ = roc_curve(y_test, y_score_clf)
    roc_auc_clf = auc(fpr_clf, tpr_clf)
    
    plt.figure(figsize = (10,10))
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.00])
    plt.plot(fpr_clf, tpr_clf, lw = 3, label = 'LogRegr ROC Curve ( area = {:.2f})'.format(roc_auc_clf))
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)
    plt.title('ROC Curve', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.show()


### ---- Decision Tree Classifier ---- ###

clf = DecisionTreeClassifier(random_state = 0).fit(X_train_scaled, y_train)

grid_values = {'max_depth' : [2, 3, 4, 5], 'min_samples_split' : [5, 10, 15, 20]}
grid_clf = GridSearchCV(clf, param_grid = grid_values, cv = 5, scoring = 'precision')
grid_clf.fit(X_train_scaled, y_train)
print('Grid Search best parameter : {0}'.format(grid_clf.best_params_))
    
clf = DecisionTreeClassifier(max_depth = grid_clf.best_params_['max_depth'], min_samples_split = grid_clf.best_params_['min_samples_split'], random_state = 0).fit(X_train_scaled, y_train)

y_predict = clf.predict(X_test_scaled)
y_score_clf = clf.predict_proba(X_test_scaled)[:,1]


print('\n')
print('Acccuracy of Decision Tree classifier on training set', clf.score(X_train_scaled, y_train))
print('Accuracy of Decision Tree classifier on test set', clf.score(X_test_scaled, y_test))
print('Precision', precision_score(y_test, y_predict))
print('Recall', recall_score(y_test, y_predict))
print('F1-score', f1_score(y_test, y_predict))

fpr_clf, tpr_clf, _ = roc_curve(y_test, y_score_clf)
roc_auc_clf = auc(fpr_clf, tpr_clf)
    
plt.figure(figsize = (10,10))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.00])
plt.plot(fpr_clf, tpr_clf, lw = 3, label = 'DecisionTree ROC Curve ( area = {:.2f})'.format(roc_auc_clf))
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('ROC Curve', fontsize = 16)
plt.legend(fontsize = 16)
plt.show()





### ---- Random Forest Classifier ---- ###

clf = RandomForestClassifier(random_state = 0).fit(X_train_scaled, y_train)

grid_values = {'n_estimators' : [10, 20, 30, 40, 50], 'max_depth' : [2, 3, 4, 5], 'min_samples_split' : [5, 10, 15, 20]}
grid_clf = GridSearchCV(clf, param_grid = grid_values, cv = 5, scoring = 'precision')
grid_clf.fit(X_train_scaled, y_train)
print('Grid Search best parameter : {0}'.format(grid_clf.best_params_))

clf = RandomForestClassifier(n_estimators = grid_clf.best_params_['n_estimators'], max_depth = grid_clf.best_params_['max_depth'], min_samples_split = grid_clf.best_params_['min_samples_split'], random_state = 0).fit(X_train_scaled, y_train)
               
y_predict = clf.predict(X_test_scaled)
y_score_clf = clf.predict_proba(X_test_scaled)[:,1]


print('\n')
print('Acccuracy of Random Forest classifier on training set', clf.score(X_train_scaled, y_train))
print('Accuracy of Random Forest classifier on test set', clf.score(X_test_scaled, y_test))
print('Precision', precision_score(y_test, y_predict))
print('Recall', recall_score(y_test, y_predict))
print('F1-score', f1_score(y_test, y_predict))

fpr_clf, tpr_clf, _ = roc_curve(y_test, y_score_clf)
roc_auc_clf = auc(fpr_clf, tpr_clf)
    
plt.figure(figsize = (10,10))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.00])
plt.plot(fpr_clf, tpr_clf, lw = 3, label = 'RandomForest ROC Curve ( area = {:.2f})'.format(roc_auc_clf))
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('ROC Curve', fontsize = 16)
plt.legend(fontsize = 16)
plt.show()       





### ---- Gradient Boosting Classifier ---- ###

clf = GradientBoostingClassifier(random_state = 0).fit(X_train_scaled, y_train)

grid_values = {'n_estimators' : [10, 20, 30, 40, 50], 'learning_rate' : [0.01, 0.1, 1.0], 'max_depth' : [2, 3, 4, 5], 'min_samples_split' : [5, 10, 15, 20]}
grid_clf = GridSearchCV(clf, param_grid = grid_values, cv = 5, scoring = 'precision')
grid_clf.fit(X_train_scaled, y_train)
print('Grid Search best parameter : {0}'.format(grid_clf.best_params_))

clf = GradientBoostingClassifier(n_estimators = grid_clf.best_params_['n_estimators'], learning_rate = grid_clf.best_params_['learning_rate'], max_depth = grid_clf.best_params_['max_depth'], min_samples_split = grid_clf.best_params_['min_samples_split'], random_state = 0).fit(X_train_scaled, y_train)

y_predict = clf.predict(X_test_scaled)
y_score_clf = clf.predict_proba(X_test_scaled)[:,1]


print('\n')
print('Acccuracy of Gradient Boosting classifier on training set', clf.score(X_train_scaled, y_train))
print('Accuracy of Gradient Boosting classifier on test set', clf.score(X_test_scaled, y_test))
print('Precision', precision_score(y_test, y_predict))
print('Recall', recall_score(y_test, y_predict))
print('F1-score', f1_score(y_test, y_predict))

fpr_clf, tpr_clf, _ = roc_curve(y_test, y_score_clf)
roc_auc_clf = auc(fpr_clf, tpr_clf)
    
plt.figure(figsize = (10,10))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.00])
plt.plot(fpr_clf, tpr_clf, lw = 3, label = 'GradientBoosting ROC Curve ( area = {:.2f})'.format(roc_auc_clf))
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('ROC Curve', fontsize = 16)
plt.legend(fontsize = 16)
plt.show()       

    

