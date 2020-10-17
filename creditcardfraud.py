
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Loading the dataset
df = pd.read_csv('creditcard.csv')

# 'Class' = 1 - fraudulent transactions
fraud = df[df['Class'] == 1]
# 'Class' = 0 - non - fraudulent or normal transactions
normal = df[df['Class'] == 0] 


print(fraud.shape,normal.shape)


# Finding correlations - that is determining how different features affect the Class (Fraud or not)

corr = df.corr()
corr

# heatmap - uses color in order to communicate a value to the reader.
corr = df.corr()
plt.figure(figsize = (12,10))
heat = sns.heatmap(data = corr)
plt.title('Heatmap of Correlation')


# From the heatmap, we get an idea of to what degree different features contribute to the transaction being fraudulent or not.
# skewness
# Finding the skewness of the features 
# to ensure that they are not much deviated from the Gaussian distribution
# As presence of much skewness in features may violate our training algo assumptions
skew_ = df.skew()
skew_

# Data Preprocessing

# 1. Scaling Amount and Time

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler2 = StandardScaler()
# scaling the time column
scaled_time = scaler.fit_transform(df[['Time']])
flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]
scaled_time = pd.Series(flat_list1)

# scaling the amount column
scaled_amount = scaler2.fit_transform(df[['Amount']])
flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]
scaled_amount = pd.Series(flat_list2)

# concatenating newly created scaled columns with original df
df = pd.concat([df, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)
# viewing a random sample of items from an axis of the object
df.sample(5)

# dropping old (unscaled) amount and time columns
df.drop(['Amount', 'Time'], axis = 1, inplace = True)

# manual train test split using numpy's random.rand
mask = np.random.rand(len(df)) < 0.9
train = df[mask]
test = df[~mask]
print('Train Shape: {}\nTest Shape: {}'.format(train.shape, test.shape))

# resetting the indices 
train.reset_index(drop = True, inplace = True)
test.reset_index(drop = True, inplace = True)


# Creating a subsample data set with balanced class distributions

# how many random samples from normal transactions do we need?
no_of_frauds = train.Class.value_counts()[1]
print('There are {} fraudulent transactions in the train data.'.format(no_of_frauds))

# storing the non - fraudulent and fraudulent transactions in the train data
non_fraud = train[train['Class'] == 0]
fraud = train[train['Class'] == 1]

# randomly selecting 449 random non - fraudulent transactions
selected = non_fraud.sample(no_of_frauds)
selected.shape

# printing the first 5 selected items
selected.head()

# resetting the indices
selected.reset_index(drop = True, inplace = True)
fraud.reset_index(drop = True, inplace = True)

# concatenating both (449 fraud and non - fraud transactions) 
# into a subsample data set with equal class distribution
subsample = pd.concat([selected, fraud])
len(subsample) # 449 (fraud) + 449 (non - fraud) = 898

# shuffling our data set
subsample = subsample.sample(frac = 1).reset_index(drop = True)

subsample.head(10)

# 4. Visualisation of fraud and non - fraud classes in subsample dataset created

new_counts = subsample.Class.value_counts()
plt.figure(figsize=(8,6))
sns.barplot(x=new_counts.index, y=new_counts)
plt.title('Count of Fraudulent vs. Non-Fraudulent Transactions In Subsample')
plt.ylabel('Count')
plt.xlabel('Class (0:Non-Fraudulent, 1:Fraudulent)')

# We find that, the distribution of fraud and non - fraud transactions is balanced in our subsample dataset unlike the original highly unbalanced dataset

# taking a look at correlations once more
corr = subsample.corr()
corr = corr[['Class']]
corr

# negative correlations smaller than -0.5
corr[corr.Class < -0.5]

# positive correlations greater than 0.5
corr[corr.Class > 0.5]

# 5. Extreme Outlier Removal

# Only removing the extreme outliers
Q1 = subsample.quantile(0.25)
Q3 = subsample.quantile(0.75)
IQR = Q3 - Q1

df2 = subsample[~((subsample < (Q1 - 2.5 * IQR)) |(subsample > (Q3 + 2.5 * IQR))).any(axis=1)]

len_after = len(df2)
len_before = len(subsample)
len_difference = len(subsample) - len(df2)
print('We reduced our data size from {} transactions by {} transactions to {} transactions.'.format(len_before, len_difference, len_after))


# 6. Dimensionality Reduction

from sklearn.manifold import TSNE

X = df2.drop('Class', axis=1)
y = df2['Class']

# t-SNE
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

# t-SNE scatter plot
import matplotlib.patches as mpatches

f, ax = plt.subplots(figsize=(24,16))

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax.set_title('t-SNE', fontsize=14)

ax.grid(True)

ax.legend(handles=[blue_patch, red_patch])

# Classification Algorithms

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train
from sklearn.ensemble import RandomForestClassifier

# visualizing RF
RandomForest_model = RandomForestClassifier(n_estimators = 10)

# Train
RandomForest_model.fit(X_train, y_train)

# testing the model 
RandomForest_predict = RandomForest_model.predict(X_test)

# accuracy
from sklearn.metrics import accuracy_score
rf_accuracy = accuracy_score(y_test, RandomForest_predict)
print("Model has a Score Accuracy: {:.3%}".format(rf_accuracy))

import pandas as pd
import pickle

# saving model to disk
pickle.dump(RandomForest_model, open('model.pkl', 'wb'))

# Loading model to compare results
model = pickle.load(open('model.pkl', 'rb'))


