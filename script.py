import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn import metrics
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import chi2_contingency


X = pd.read_csv('mer_de-id.csv')

y = X['label']
X_study_id = X.drop(['Infective Endocarditis? \n1.Yes\n0.No', 'Days of therapy'],  axis = 1)

X = X.drop(['study_id', 'label', 'Infective Endocarditis? \n1.Yes\n0.No', 'Days of therapy'],  axis = 1)

np.random.seed(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# select the classifer by uncommenting 
#clf = svm.SVC(kernel='rbf')
#clf = svm.SVC(kernel='linear')
#clf = LogisticRegression()
#clf = RandomForestClassifier()
clf = xgb.XGBClassifier()

# Define the parameter grid for the number of estimators
param_grid = {'n_estimators': [100, 200, 300, 400, 500]}

# Define the hyperparameter grid for C penaltiy in linear SVM
#param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# Split the data into training and validation sets
X_tr, X_v, y_tr, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Create the GridSearchCV object

# grid search 5-folds cv 
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
# Fit the GridSearchCV object to the training data
grid_search.fit(X_tr, y_tr)

# Get the best estimator and its corresponding parameters
best_estimator = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the best estimator on the validation set
validation_accuracy = best_estimator.score(X_v, y_v)

# Print the best parameters and validation accuracy
print("Best Parameters:", best_params)
print("Validation Accuracy:", validation_accuracy)

#Select the classifier for the boostrap evaluation with the optimal hyperparamter selected via 5-fold CV
#clf = svm.SVC(kernel='rbf', C = best_params['C'])
#clf = svm.SVC(kernel='linear', C = best_params['C'])
#clf = LogisticRegression(C = best_params['C'], max_iter=1000)
#clf = RandomForestClassifier(n_estimators = best_params['n_estimators'])
clf = xgb.XGBClassifier(n_estimators = best_params['n_estimators'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# min-max scaler 
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

#X_train = X_train.iloc[:, top_10_indices]
#X_test = X_test.iloc[:, top_10_indices]

# Define the number of bootstrap iterations
n_iterations = 1000

# Create an array to store the accuracies
accuracies = []
baccs = []
specificities = []
sensitivities = []
f1_scores = []
misclassified_indices = []
misclassified_matrix = []
# Confusion matrix 
aggregate_cm = np.zeros((2, 2))

# Perform bootstrapping and evaluate classifier performance
for _ in range(n_iterations):
    # Create a bootstrap sample by sampling with replacement
    X_boot, y_boot = resample(X_train, y_train, replace=True)
    # Fit the classifier on the bootstrap sample
    clf.fit(X_boot, y_boot)


    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy and store it
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)
    f1_scores.append(f1_score(y_test, y_pred))
    baccs.append(balanced_accuracy_score(y_test, y_pred))
    sensitivity = metrics.recall_score(y_test, y_pred)
    sensitivities.append(sensitivity)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    specificities.append(specificity)
    cm = confusion_matrix(y_test, y_pred)
    aggregate_cm += cm

    # Find misclassified instances and store their indices and class labels
    misclassified = np.where(y_pred != y_test)[0]
    misclassified_indices.extend(misclassified)

# Plot the aggregated confusion matrix
labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
categories = ['0', '1']
fig, ax = plt.subplots()
im = ax.imshow(aggregate_cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(aggregate_cm.shape[1]),
       yticks=np.arange(aggregate_cm.shape[0]),
       xticklabels=categories, yticklabels=categories,
       title='Aggregated Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)
ax.grid(False)

thresh = aggregate_cm.max() / 2.
for i in range(aggregate_cm.shape[0]):
    for j in range(aggregate_cm.shape[1]):
        ax.text(j, i, format(aggregate_cm[i, j], '.0f'),
                ha="center", va="center",
                color="white" if aggregate_cm[i, j] > thresh else "black")

plt.savefig('aggregated_cm.png')
#plt.show()

# performence metrics and 95% CI 

m_name = 'baccs'
m = baccs

# Calculate the lower and upper bounds for the confidence interval
lower_bound = np.percentile(m, 2.5)
upper_bound = np.percentile(m, 97.5)

# Calculate the mean balanced accuracy
mean_m = np.mean(m)
#mean_baccs = np.mean(baccs)
# Print the mean balanced accuracy and confidence interval
print('mean ' + m_name + ' :', mean_m)
print("Confidence Interval (95%):", lower_bound, "-", upper_bound)


m_name = 'sensitivities'
m = sensitivities

# Calculate the lower and upper bounds for the confidence interval
lower_bound = np.percentile(m, 2.5)
upper_bound = np.percentile(m, 97.5)

# Calculate the mean sensitivities
mean_m = np.mean(m)
# Print the mean sensitivity and confidence interval
print('mean ' + m_name + ' :', mean_m)
print("Confidence Interval (95%):", lower_bound, "-", upper_bound)

m_name = 'specificities'
m = specificities

# Calculate the lower and upper bounds for the confidence interval
lower_bound = np.percentile(m, 2.5)
upper_bound = np.percentile(m, 97.5)

# Calculate the mean 
mean_m = np.mean(m)
# Print the mean specificity and confidence interval
print('mean ' + m_name + ' :', mean_m)
print("Confidence Interval (95%):", lower_bound, "-", upper_bound)

m_name = 'f1_scores'
m = f1_scores

# Calculate the lower and upper bounds for the confidence interval
lower_bound = np.percentile(m, 2.5)
upper_bound = np.percentile(m, 97.5)

# Calculate the mean f1_score
mean_m = np.mean(m)

# Print the mean f1_score and confidence interval
print('mean ' + m_name + ' :', mean_m)
print("Confidence Interval (95%):", lower_bound, "-", upper_bound)


# save the results of misclassification 
filtered_df = X_study_id[X_study_id.index.isin(y_test.index)]
filtered_df
misclassified_indices
header = filtered_df.columns
def occurrence(numbers):
    occurrences = {}
    for num in numbers:
        if num in occurrences:
            occurrences[num] += 1
        else:
            occurrences[num] = 1
    return occurrences

result = occurrence(misclassified_indices)
result
filtered_df
filtered_df.to_csv('test_set.csv', index=True)
type(result)

ind = list(range(len(y_test)))

result = pd.Series(result)
result_reindexed = result.reindex(ind)


#concatenated_series
sorted_y_test = y_test.sort_index()
array1 = np.asarray(result_reindexed).reshape(-1, 1)
array = sorted_y_test.reset_index().values
array3 = np.concatenate((array,array1), axis = 1)
array3[np.isnan(array3)] = 0
array3 = array3.astype(int)
filtered = filtered_df.values

np.savetxt('misclassifed.csv', array3, delimiter=',', fmt='%d')

df3 = pd.DataFrame(array3)
df4 = pd.DataFrame(filtered)
df_concatenated = pd.concat([df4, df3], axis=1)
df_concatenated

header2 = ['index', 'label', 'misclassified_occur']
new_header = header.append(pd.Index(header2))
new_header
df_concatenated.columns = new_header
df_concatenated
df_concatenated.to_csv('test_set_results.csv', index = False)


# Heatmap on the entire data between pair of features with chi-square test p-values

# Combine the X and target dataframes
data = pd.concat([X, target], axis=1)

# Remove features with no variations
data = data.loc[:, data.nunique() > 1]

# Create a p-value matrix
p_value_matrix = pd.DataFrame(index=data.columns, columns=data.columns)

# Calculate p-values for each feature combination
for i, feature1 in enumerate(data.columns):
    for j, feature2 in enumerate(data.columns):
        contingency_table = pd.crosstab(index=data[feature1], columns=data[feature2])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        p_value_matrix.iloc[i, j] = p_value

# Rename columns and index with f_0, f_1, f_2, ...
new_index = ['f_' + str(i) for i in range(len(data.columns)-1)]  # Subtract 1 for excluding the target
p_value_matrix = p_value_matrix.rename(index=dict(zip(data.columns, new_index)))
p_value_matrix = p_value_matrix.rename(columns=dict(zip(data.columns, new_index)))

# Create heatmap matrix with modified color scheme and smaller font size for the legend
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(p_value_matrix.astype(float), annot=True, fmt=".2f", cmap='RdBu_r', cbar=True, ax=ax, annot_kws={'size': 7},
            cbar_kws={'shrink': 0.8})
plt.title('P-value Heatmap Matrix')
plt.xticks(ticks=range(len(data.columns)-1), labels=p_value_matrix.columns[:-1], fontsize=10, ha='center')
plt.yticks(ticks=range(len(data.columns)-1), labels=p_value_matrix.index[:-1], fontsize=10)
plt.xlabel('target')
plt.savefig('heatmap_chi_square.png')
#plt.show()





