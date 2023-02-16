import pandas as pd
import xlrd
import numpy as np

xls = pd.ExcelFile(r"WeatherData.xls")
df = xls.parse(0)

display(df)
df.describe()

#Remove Outliers
# IQR
import seaborn as sns
sns.boxplot(data=df, orient="h")
def remove_outliers(df, columns, multiplier=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df
outlier_columns=['MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Temp9am','Temp3pm','Pressure3pm']
df=remove_outliers(df,outlier_columns)
sns.boxplot(data=df, orient="h")




#Changing Rain Tommorow and Today to Binary
df["RainTomorrow"].replace(['No', 'Yes'],[0, 1], inplace=True)
df["RainToday"].replace(['No', 'Yes'],[0, 1], inplace=True)
df["Location"].replace(["Region1","Region2","Region3","Region4","Region5","Region6","Region7","Region8","Region9","Region10","Region11","Region12"]
                       ,[1,2,3,4,5,6,7,8,9,10,11,12],inplace=True)
df.describe()

#Adding Constants(min value) instead of Null Values for Numeric Columns
#Dropping Rows with Null Categorical Value

values={"RainToday":0,"RainTomorrow":0,"MinTemp":-4.8,"MaxTemp":6.8,"Temp9am":0.3,"Temp3pm":6.4,"Cloud9am": 0,"Cloud3pm": 0,"Rainfall": 2.696772,"WindGustSpeed": 7,"WindSpeed9am":0,"WindSpeed3pm":0,"Pressure9am": 980.0,"Pressure3pm": 980.0,'Humidity9am': 1,'Humidity3pm': 1}
df.fillna(value=values,inplace=True)
#df.dropna(subset=["RainTomorrow"],inplace=True)
df.describe()
display(df)
sns.boxplot(data=df, orient="h")

#Feature Scaling


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer

columns_for_scaling=['MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm']

scaler = StandardScaler().set_output(transform="pandas")


from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.feature_selection import SelectKBest, mutual_info_classif

feature_cols=['RainToday','MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed3pm','Humidity3pm','Pressure3pm','Cloud3pm','Temp3pm']
X=df[feature_cols]
Y=df['RainTomorrow']
X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=1)

X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test = scaler.fit_transform(X_test)

selector = SelectKBest(mutual_info_classif, k=7)
X_new = selector.fit_transform(X_train, y_train)

selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)

X=df[selected_features]
Y=df['RainTomorrow']
X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=1)

X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test = scaler.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

max_depth_list = [3, 5, 7, 9, 11, 13, 15]

# Initialize a list to store the accuracy scores
accuracy_scores = []

# Iterate through each max_depth value
for max_depth in max_depth_list:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    # Store the accuracy score
    accuracy_scores.append(accuracy)

# Find the max_depth value with the highest accuracy score
best_max_depth = max_depth_list[accuracy_scores.index(max(accuracy_scores))]

data = {'Max Depth': max_depth_list, 'Accuracy Score': accuracy_scores}
dfMaxDepth = pd.DataFrame(data)
display(dfMaxDepth)

from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
import math
model = DecisionTreeClassifier(max_depth=best_max_depth)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_test)
#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test,y_pred))
print("RMSE:",math.sqrt(mean_squared_error(y_test,y_pred)))
print("R Squared: ", r2_score(y_test,y_pred))

from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import zero_one_loss
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        model, X_train.values, y_train.values, X_test.values, y_test.values,
        loss='0-1_loss',
        random_seed=123)

print('Average expected loss--After pruning: %.3f' % avg_expected_loss)
print('Average bias--After pruning: %.3f' % avg_bias)
print('Average variance--After pruning: %.3f' % avg_var)
print('Sklearn 0-1 loss--After pruning: %.3f' % zero_one_loss(y_test.values,y_pred))

import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=X_train.columns, class_names=y_train.unique().astype(str))
plt.show()
plt.savefig("WeatherDecisionTree.png")

#Logistic Regression
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
logisticRegression = LogisticRegression(max_iter=150)
logisticRegression.fit(X_train, y_train)
logisticPredictions = logisticRegression.predict(X_test)

#evaluate the model
print(classification_report(y_test,logisticPredictions))
print("RMSE:",sqrt(mean_squared_error(y_test,logisticPredictions)))
print("R Squared: ", r2_score(y_test,logisticPredictions))

from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import zero_one_loss
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        logisticRegression, X_train.values, y_train.values, X_test.values, y_test.values,
        loss='0-1_loss',
        random_seed=123)

print('Average expected loss--After pruning: %.3f' % avg_expected_loss)
print('Average bias--After pruning: %.3f' % avg_bias)
print('Average variance--After pruning: %.3f' % avg_var)
print('Sklearn 0-1 loss--After pruning: %.3f' % zero_one_loss(y_test.values,y_pred))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

fpr, tpr, thresholds = roc_curve(y_test, logisticPredictions)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])

#Neural Network
import numpy as np
from math import sqrt
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score
neuralNetwork = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(12,), random_state=1,max_iter=1000)
neuralNetwork.fit(X_train, y_train)

NNPredictions = neuralNetwork.predict(X_test)
print(classification_report(y_test,NNPredictions))
print("RMSE",sqrt(mean_squared_error(y_test,NNPredictions)))
print("R Squared: ", r2_score(y_test,NNPredictions))

from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import zero_one_loss
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        neuralNetwork, X_train.values, y_train.values, X_test.values, y_test.values,
        loss='0-1_loss',
        random_seed=123)

print('Average expected loss--After pruning: %.3f' % avg_expected_loss)
print('Average bias--After pruning: %.3f' % avg_bias)
print('Average variance--After pruning: %.3f' % avg_var)
print('Sklearn 0-1 loss--After pruning: %.3f' % zero_one_loss(y_test.values,y_pred))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

fpr, tpr, thresholds = roc_curve(y_test, NNPredictions)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])

# K Means Clustering
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

kData = df[selected_features].to_numpy()

kmeans = KMeans(n_clusters=2)

# Fit the model to the data
kmeans.fit(kData)

# Predict the cluster labels for each sample
labels = kmeans.predict(kData)
print(classification_report(df['RainTomorrow'], labels))
print("RMSE: ", sqrt(mean_squared_error(df['RainTomorrow'], labels)))
print("R Squared: ", r2_score(df['RainTomorrow'], labels))

# Getting unique labels

u_labels = np.unique(labels)

# plotting the results:

for i in u_labels:
    plt.scatter(kData[labels == i, 0], kData[labels == i, 1], label=i)
plt.legend()
plt.show()

from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import zero_one_loss
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        kmeans, X_train.values, y_train.values, X_test.values, y_test.values,
        loss='0-1_loss',
        random_seed=123)

print('Average expected loss--After pruning: %.3f' % avg_expected_loss)
print('Average bias--After pruning: %.3f' % avg_bias)
print('Average variance--After pruning: %.3f' % avg_var)
print('Sklearn 0-1 loss--After pruning: %.3f' % zero_one_loss(y_test.values,y_pred))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

fpr, tpr, thresholds = roc_curve(df['RainTomorrow'], labels)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])

# SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

svm = SVC(kernel="rbf", C=10,gamma=0.1)
svm.fit(X_train,y_train)

svmPredictions = svm.predict(X_test)
print(classification_report(y_test,svmPredictions))
print("RMSE:" ,sqrt(mean_squared_error(y_test,svmPredictions)))
print("R Squared: ", r2_score(y_test,svmPredictions))

from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import zero_one_loss
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        svm, X_train.values, y_train.values, X_test.values, y_test.values,
        loss='0-1_loss',
        random_seed=123)

print('bias %.3f' % avg_bias)
print('Average variance--After pruning: %.3f' % avg_var)