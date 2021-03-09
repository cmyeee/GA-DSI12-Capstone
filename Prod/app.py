# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:11:19 2020

@author: cmyee
"""

import flask
from flask import flash
# from flask.ext.session import Session
import dill 
import numpy as np
import pandas as pd 

app = flask.Flask(__name__)

dill._dill._reverse_typemap['ClassType'] = type #Takes care of "KeyError" issue of dill while unpickling

with open('rfPipe.pkl', 'rb') as f:
    pipe = dill.load(f)

##################################
##################################
#'/' = www.mywebsite.com
#'/test' = www.mywebsite.com/test

# #NOTES
# Dont mix up non-tech and tech details
# cook up data for website - done

@app.route('/', methods=['POST', 'GET'])

# create df for staff available  


def page():
    df = pd.read_csv("Test.csv")
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':
        #print(df)
        #obtain filtered df based on employee input, then produce score
        inputs = flask.request.form
        emp_num = inputs['EmployeeNumber']
        if int(emp_num) in list(df.EmployeeNumber):
            df_search = df[df['EmployeeNumber'] == int(emp_num)]
            
            score = pipe.predict_proba(df_search.drop(columns='Attrition'))
            resign = int(score[0,1] * 100)
            notresign = int(score[0,0] * 100)
            
            df_display = df_search[['OverTime','WorkLifeBalance','YearsSinceLastPromotion', 'BusinessTravel', 'JobRole', 'JobInvolvement','EnvironmentSatisfaction', 
                                   'JobSatisfaction']]
            
            df_html = df_display.T.to_html(header=False, justify='center')
                
        else:
            #flash('The EmployeeNumber does not exist.')
            resign = 0
            notresign = 0
        
        #obtain coefficients from pipeline.
        #mycoef = pipe.named_steps['lr'].coef_
        
    else:      
        resign = 0
        notresign = 0
        emp_num = ''
        df_html = ''
        
    return flask.render_template('dataentrypage.html', resign=resign, notresign=notresign, emp_num=emp_num, df_html=df_html)

##################################
# sess = Session()

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    # app.config['SESSION_TYPE'] = 'filesystem'

    # sess.init_app(app)

    app.run(debug=True, port=5008)