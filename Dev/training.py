# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:41:17 2020

@author: cmyee
"""

# import statements
############################################
# Here are all the libraries that we need. #
############################################

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
#from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

import dill

# Global Constants
datafile_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
pickle_file_path = './rfPipe.pkl'


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []
        
    
    # convert categorical numeric scales to obj type data    
    def _convert_obj(self, df):
        
        # Dictionaries for DataCleaner
        edu_dict = { 1:'Below College', 
                     2:'College',
                     3:'Bachelor',
                     4:'Master',
                     5:'Doctor' }
            
        scale_dict = { 1:'Low',
                       2:'Medium',
                       3:'High',
                       4:'Very High' }
        
        perf_dict = { 1:'Low',
                      2:'Good',
                      3:'Excellent',
                      4:'Outstanding' }
        
        work_life_dict = { 1:'Bad',
                           2:'Good',
                           3:'Better',
                           4:'Best'}
        
        stock_dict =    { 0:'Nil',
                          1:'1stock',
                          2:'2stock',
                          3:'3stock' }
        
        df['Education_obj'] = df['Education'].map(edu_dict)
        df['JobSatisfaction_obj'] = df['JobSatisfaction'].map(scale_dict)
        df['RelationshipSatisfaction_obj'] = df['RelationshipSatisfaction'].map(scale_dict)
        df['EnvironmentSatisfaction_obj'] = df['EnvironmentSatisfaction'].map(scale_dict)
        df['JobInvolvement_obj'] = df['JobInvolvement'].map(scale_dict)
        df['PerformanceRating_obj'] = df['PerformanceRating'].map(perf_dict)
        df['WorkLifeBalance_obj'] = df['WorkLifeBalance'].map(work_life_dict)
        df['StockOptionLevel_obj'] = df['StockOptionLevel'].map(stock_dict)
        return df
    
    # drop unwanted columns (1)
    def _drop_unused1(self, df):
        for col in ['EmployeeCount', 'Over18', 'StandardHours', 'DailyRate', 'HourlyRate', 'MonthlyRate', 'JobLevel',
                    'Education','JobSatisfaction', 'RelationshipSatisfaction','EnvironmentSatisfaction','JobInvolvement',
                    'PerformanceRating','WorkLifeBalance','StockOptionLevel', 'EmployeeNumber']:
            try:
                df = df.drop(col, axis=1)
            except:
                pass
        return df
    
    # get dummy variables 
    def _get_dummies(self, df):
        df = pd.get_dummies(df)
        return df
    
    # add in missing columns to get 55 variables.  
    def _missing_col(self, df):
        col_list = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager', 'BusinessTravel_Non-Travel',
       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
       'Department_Human Resources', 'Department_Research & Development',
       'Department_Sales', 'EducationField_Human Resources',
       'EducationField_Life Sciences', 'EducationField_Marketing',
       'EducationField_Medical', 'EducationField_Other',
       'EducationField_Technical Degree', 'Gender_Female', 'Gender_Male',
       'JobRole_Healthcare Representative', 'JobRole_Human Resources',
       'JobRole_Laboratory Technician', 'JobRole_Manager',
       'JobRole_Manufacturing Director', 'JobRole_Research Director',
       'JobRole_Research Scientist', 'JobRole_Sales Executive',
       'JobRole_Sales Representative', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_No',
       'OverTime_Yes', 'Education_obj_Bachelor', 'Education_obj_Below College',
       'Education_obj_College', 'Education_obj_Doctor', 'Education_obj_Master',
       'JobSatisfaction_obj_High', 'JobSatisfaction_obj_Low',
       'JobSatisfaction_obj_Medium', 'JobSatisfaction_obj_Very High',
       'RelationshipSatisfaction_obj_High', 'RelationshipSatisfaction_obj_Low',
       'RelationshipSatisfaction_obj_Medium',
       'RelationshipSatisfaction_obj_Very High',
       'EnvironmentSatisfaction_obj_High', 'EnvironmentSatisfaction_obj_Low',
       'EnvironmentSatisfaction_obj_Medium',
       'EnvironmentSatisfaction_obj_Very High', 'JobInvolvement_obj_High',
       'JobInvolvement_obj_Low', 'JobInvolvement_obj_Medium',
       'JobInvolvement_obj_Very High', 'PerformanceRating_obj_Excellent',
       'PerformanceRating_obj_Outstanding', 'WorkLifeBalance_obj_Bad',
       'WorkLifeBalance_obj_Best', 'WorkLifeBalance_obj_Better',
       'WorkLifeBalance_obj_Good', 'StockOptionLevel_obj_1stock',
       'StockOptionLevel_obj_2stock', 'StockOptionLevel_obj_3stock',
       'StockOptionLevel_obj_Nil']
        for col in col_list:
            if col not in df.columns:
                df[col] = 0
        return df

    # drop unwanted columns after creating dummies (2)
    def _drop_unused2(self, df):
        for col in ['BusinessTravel_Non-Travel', 'Department_Human Resources','EducationField_Human Resources', 
                    'Gender_Female','JobRole_Human Resources', 'MaritalStatus_Divorced','OverTime_No', 
                    'Education_obj_Doctor','JobSatisfaction_obj_Very High', 'RelationshipSatisfaction_obj_Very High', 
                    'EnvironmentSatisfaction_obj_Very High','JobInvolvement_obj_Very High', 
                    'PerformanceRating_obj_Excellent', 'WorkLifeBalance_obj_Best', 'StockOptionLevel_obj_3stock',
                    'Attrition_No']:
            try:
                df = df.drop(col, axis=1)
            except:
                pass
        return df
    
    def transform(self, X, *args):
        X = self._convert_obj(X)
        X = self._drop_unused1(X)
        X = self._get_dummies(X)
        X = self._missing_col(X)
        X = self._drop_unused2(X)
        self.feature_names = X.columns
        return X
    
    def fit(self, X, *args):
        return self

# Get X and Y variables
def load_data():
    df = pd.read_csv(datafile_path)
    df['Attrition'] = df['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return X,y
    
# Perform feature selection using SelectKBest and RFECV
def fit_pipe(X,y):
    
    dc = DataCleaner()
    
    # Oversampling 
    ros = RandomOverSampler(random_state=42)

    # StandardScaler 
    ss = StandardScaler()
    
    # SelectKBest
    kbest = SelectKBest(k=35)
    
    # RFE
    lr_rfe = LogisticRegression(C=0.12915496650148828, max_iter=2000, penalty='l1', solver='liblinear')
    rfe = RFE(lr_rfe, n_features_to_select=25, step=1)
    
    #Logistic Regression
    lr = LogisticRegression(C=464.1588833612773, max_iter=2000, penalty='l1', solver='liblinear')
    
    #build pipeline
    pipe = Pipeline(steps = [('dc', dc), ('ros', ros), ('ss', ss), ('kbest', kbest), ('rfe',rfe), ('lr',lr)])
    pipe.fit(X,y)
    return pipe
    
#Serialization or Pickling
def serialize(pipe):
    with open(pickle_file_path, 'wb') as f:
        dill.dump(pipe, f)  

#now, LR pipe is not printed when I run the script. 
def main():
    try:
        X, y = load_data()
        pipe = fit_pipe(X,y)
        serialize (pipe)
        print ('RF pipeline is trained and serialized')
    except Exception as err:
        print (err.args)
        exit

#Program Entry Point
if __name__ == '__main__':
    main()

