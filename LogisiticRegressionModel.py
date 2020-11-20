import csv
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

#Reading in the features from clean_data.csv
data = pd.read_csv('clean_data.csv')

#Separate the values from each column
death          = data['death_yn'].values
current_status = data['current_status'].values
sex            = data['sex'].values
age_group      = data['age_group'].values
race_ethnicity = data['Race and ethnicity (combined)'].values
hospitalized   = data['hosp_yn'].values
icu            = data['icu_yn'].values
med_condition  = data['medcond_yn'].values

#Preprocessing
death_fix = []
for x in death:
    if x == 'Yes':
        death_fix.append(1)
    elif x == 'No':
        death_fix.append(0)
death = death_fix

current_status_fix = []
for x in current_status:
    if x == 'Laboratory-confirmed case':
        current_status_fix.append(1)
    else:
        current_status_fix.append(0)
current_status = current_status_fix

sex_fix = []
for x in sex:
    if x == 'Male':
        sex_fix.append(0)
    elif x == 'Female':
        sex_fix.append(1)
    elif x == 'Other':
        sex_fix.append(2)
sex = sex_fix

age_group_fix = []
age_group_type = []
for x in age_group:
    if x not in age_group_type:
        age_group_type.append(x)
    age_group_fix.append(age_group_type.index(x))
age_group = age_group_fix

race_ethnicity_fix = []
race_ethnicity_type = []
for x in race_ethnicity:
    if x not in race_ethnicity_type:
        race_ethnicity_type.append(x)
    race_ethnicity_fix.append(race_ethnicity_type.index(x))
race_ethnicity = race_ethnicity_fix

hospitalized_fix = []
for x in hospitalized:
    if x == 'Yes':
        hospitalized_fix.append(1)
    elif x == 'No':
        hospitalized_fix.append(0)
hospitalized = hospitalized_fix

icu_fix = []
for x in icu:
    if x == 'Yes':
        icu_fix.append(1)
    elif x == 'No':
        icu_fix.append(0)
icu = icu_fix

med_condition_fix = []
for x in med_condition:
    if x == 'Yes':
        med_condition_fix.append(1)
    elif x == 'No':
        med_condition_fix.append(0)
med_condition = med_condition_fix

#Matrix of ones
row_split    = math.floor(len(death)*0.8)
length_train = len(death[0:row_split])
length_test  = len(death[row_split:len(death) - 1])
x0 = np.ones(len(death))
l = len(death)

#Creating the array of features and transposing
features = ['current_status', 'sex', 'age_group', 'Race and ethnicity (combined)', \
            'hosp_yn', 'icu_yn', 'medcond_yn']

X = np.array([current_status, sex, age_group, race_ethnicity, hospitalized, icu, med_condition]).T
df = pd.DataFrame(data=X, columns=features)
X = df.values
print(X)

X_train = np.array([x0[0:row_split], current_status[0:row_split], sex[0:row_split], \
                    age_group[0:row_split], race_ethnicity[0:row_split], hospitalized[0:row_split], \
                    icu[0:row_split], med_condition[0:row_split]]).T
X_test = np.array([x0[row_split:l-1], current_status[row_split:l-1], sex[row_split:l-1], \
                   age_group[row_split:l-1], race_ethnicity[row_split:l-1], \
                   hospitalized[row_split:l-1], icu[row_split:l-1], med_condition[row_split:l-1]]).T

#Creating the y array
y = np.array(death)

#Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y[0:row_split])

y_pred = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y[row_split:l-1])

index = row_split
t_p = 0
t_n = 0
f_p = 0
f_n = 0
for i in y_pred:
    print(i, end=" ")
    if i == y[index]:
        if i == 1:
            t_p += 1
        else:
            t_n += 1
    else:
        if i == 1:
            f_p += 1
        else:
            f_n += 1
    index += 1
print("\n")
print(score)
weights = logisticRegr.coef_[0]
weights = np.delete(weights, 0)
print("Weights: ", weights)
print("True Positive: ", t_p, "True Negative: ", t_n)
print("False Positive: ", f_p, "False Negative: ", f_n)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])

print()
print("How much of our variance is explained?")
print(pca.explained_variance_ratio_)
print()
print()

print("Which features matter most?")
print(abs(pca.components_))
