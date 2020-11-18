import csv
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

class MultipleLinearRegression(object):
    def __init__(self, epochs, alpha):
        self.coefficients = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.epochs = epochs
        self.alpha = alpha

    def train(self, X, y, length):
        cost_list = []
        coeff_list = []
        prediction_list = []

        cost_list.append(1e10)
        count = 0
        while(count < self.epochs):
            # Calculating cost
            prediction = self.predict(X)
            error = prediction - y
            cost = 1/(2*length) * np.dot(error.T, error)
            cost_list.append(cost)

            #updating coefficients
            self.coefficients = self.coefficients - (self.alpha * (1/length) * np.dot(X.T, error))
            coeff_list.append(self.coefficients)

            if cost_list[count]-cost_list[count+1] < 1e-9:
                print('Ended on count: ', count)
                count = self.epochs

            count += 1
        cost_list.pop(0)
        return cost_list, coeff_list

    def predict(self, X):
        return np.dot(X, self.coefficients)

    def r2_score(self, Y, Y_pred):
        mean_y = np.mean(Y)
        ss_tot = sum((Y - mean_y) ** 2)
        ss_res = sum((Y - Y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

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
X_train = np.array([x0[0:row_split], current_status[0:row_split], sex[0:row_split], \
                    age_group[0:row_split], race_ethnicity[0:row_split], hospitalized[0:row_split], \
                    icu[0:row_split], med_condition[0:row_split]]).T
X_test = np.array([x0[row_split:l-1], current_status[row_split:l-1], sex[row_split:l-1], \
                   age_group[row_split:l-1], race_ethnicity[row_split:l-1], \
                   hospitalized[row_split:l-1], icu[row_split:l-1], med_condition[row_split:l-1]]).T

#Creating the y array
y = np.array(death)
#for i in y:
#    print(i)

#Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)

#Learning rate and epochs
alpha = 0.0025
epochs = 3000

#Creating the learning model and training it
model = MultipleLinearRegression(epochs, alpha)
costs, coeffs = model.train(X_train, y[0:row_split], length_train)
y_pred = model.predict(X_test)
r2_score = model.r2_score(y[row_split:l-1], y_pred)

print(model.coefficients)
#print(model.predict(X_test))
print(r2_score)
