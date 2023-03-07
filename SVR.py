# -*- coding: utf-8 -*-
"""
Created on Sun May 15 18:59:15 2022

@author: Tianjiao Liu
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from math import*
import PySimpleGUI as sg
count = range(100)
for i, item in enumerate(count):
    sg.one_line_progress_meter('Real-time progress bar', i + 1, len(count), '-key-')

#Define a function in the way of def, function () 1 is the function name, the parameters can be passed in parentheses, I do not pass here
def function1(): #Calculate the function of the first paragraph
    data= pd.read_csv('S1 Input train data.csv') # logging data
    AC = data.values[:832, 2:3]
    SP = data.values[:832, 3:4]

    LLD = data.values[:832, 5:6]
    LLS = data.values[:832, 6:7] 

#Training feature data
    X_train=np.concatenate((AC,SP,LLD,LLS),axis=1)
#Training target data
    S2 = data.values[:832, 1:2]
      
    #Here we give the kernel function the model needs and the range of parameters for c and gamma
    param_grid = [
      {'C': [0.001,0.1,1, 10, 100, 1000,10000], 'gamma': [0.0001,0.001,0.01,0.1,1,10,100,1000], 'kernel': ['rbf']},
    ]
    #SVR ( ) model training
    svm_model = SVR()
    #Optimize models with grid search and 5-fold cross-validation to find optimal parameters
    rbf_svr = GridSearchCV(svm_model, param_grid, cv=5)
    rbf_svr.fit(X_train, S2)
    #Get the best model
    best_model1 = rbf_svr.best_estimator_
    #Parameters and Kernel Functions of the Best Model for Print Output
    print(rbf_svr.best_params_)
   
    return best_model1    #This is the model to calculate the second return.
    
    

  
if __name__ == '__main__':
    #915 ~ 1158.3m This is the predicted data to be passed into the model
    datasetFilename_1 = 'S2 Input dataset for predict S2.CSV'
    test_x_1 = pd.read_csv(datasetFilename_1, usecols=['AC','SP','LLD','LLS'])
    test_x_1=test_x_1.loc[2:19270,['AC','SP','LLD','LLS']]
    #Call the first function, return the rbf _ svr1 model, and pass in the data you want to predict
    rbf_svr1= function1()  
    dataDict1 = test_x_1.to_dict()
    predictResult1=rbf_svr1.predict(test_x_1) 
    dataDict1['Predicted S2'] = pd.Series(predictResult1,index=list(range(2, 19269))).round(5)
    data1 = pd.DataFrame(dataDict1)
    data1.to_excel(r'Predicted_S2_all_SVR.xlsx',sheet_name='Predicted',index=True,index_label="index")

    # Suppose this code section takes 0.05 s
    time.sleep()
    
 
