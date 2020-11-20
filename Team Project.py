# Team Project
# Bennett Miller
# Griffin Lumb
# Logan Courtney
# Terryl Dodson

import warnings
import numpy as np
import pandas as pd
import sys
import sklearn as sk
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv("clean_data.csv", usecols=[i for i in range(3,11)])

training_data, test_data = np.split(data, [int(.9 * len(data))])


training_y = training_data['death_yn']
test_y = test_data['death_yn']

# Encode target variables to boolean
pd.set_option("display.max_columns", 100)
for i in range(0, len(training_y)):
    if training_y[i] == "No":
        training_y[i] = 0
    else:
        training_y[i] = 1

for i in range(221781, len(test_y)+221781):
    if test_y[i] == "No":
        test_y[i] = 0
    else:
        test_y[i] = 1

del training_data['death_yn']
del test_data['death_yn']


x_training = training_data
x_test = test_data
x_test['age_group'] = x_test['age_group'].replace('Unknown', "20 - 29 Years")
x_test.loc[x_test.index[-1], 'age_group'] = "70 - 79 Years"

# Encode training data as boolean/splits up nominal multivariate categorical variables as multiples
x_training = pd.get_dummies(x_training, drop_first=True)
x_test = pd.get_dummies(x_test, drop_first=True)
features = list(x_training.columns)
print(features)

# as ints
training_y = training_y.astype('int')
test_y = test_y.astype('int')


# train decision tree on training data
clf = tree.DecisionTreeClassifier(max_depth=4)
fit = clf.fit(x_training, training_y)
tree.export_graphviz(fit, out_file="tree.dot", feature_names=features, class_names=True)
print(clf.get_depth())

# Predict test data with DT classifier
y_predict = clf.predict(x_test)
print("Accuracy = " + str(accuracy_score(test_y, y_predict)))

# Confusion Matrix
print(pd.DataFrame(
    confusion_matrix(test_y, y_predict),
    columns = ['Predicted Dead', 'Predicted Survival'],
    index = ['True Dead', 'True Survival']
))



