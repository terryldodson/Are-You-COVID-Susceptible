import csv
import numpy as np
import pandas as pd
import math

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
