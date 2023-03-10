# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:51:50 2022

@author: Tianjiao Liu
"""
import numpy as np
from sklearn.linear_model import BayesianRidge
import pandas as pd
def function1():  #Calculate the function of the second section
    from sklearn.model_selection import GridSearchCV 
    data= pd.read_csv('S1 Input train data.csv') # Well logging data
    AC = data.values[:832, 2:3]
    SP = data.values[:832, 3:4]
    LLD = data.values[:832, 5:6]
    LLS = data.values[:832, 6:7] 

#Training target data
    S2 = data.values[:832, 1:2]
    X_train=np.concatenate((AC,SP,LLD,LLS),axis=1)
    #Here is the parameter range for the max_depth/n_estimators/max_features/min_samples_split that the model requires
    param_grid = {
        'alpha_1':[1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1e-00], 
        'alpha_2':[1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1e-00], 
        'compute_score':['False'], 
        'copy_X':['True'],
        'fit_intercept':['True'], 
        'lambda_1':[1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1e-00], 
        'lambda_2':[1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1e-00], 
        'n_iter':[100,200,300,400],
        'normalize':[False],
        'tol':[0.001], 
        'verbose':['False']
    } 
  
    #RandomForestRegressor ( ) model training
    rfc=BayesianRidge()
    #Optimize models with grid search and 5-fold cross-validation to find optimal parameters
    rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    rfc_cv.fit(X_train, S2)
    #Get the best model
    best_model1 = rfc_cv.best_estimator_
    #Parameters and Kernel Functions of the Best Model for Print Output
    print(rfc_cv.best_params_)
   
    return best_model1    #This is the model to calculate the second return.



if __name__ == '__main__':
    #915~1158.3m   This is the prediction data to pass in the model
    datasetFilename_1 = 'S2 Input dataset for predict S2.CSV'
    test_x_1 = pd.read_csv(datasetFilename_1, usecols=['AC','SP','LLD','LLS'])
    test_x_1=test_x_1.loc[2:19270,['AC','SP','LLD','LLS']]
    #Call the first function, return the rbf _ svr1 model, and pass in the data you want to predict
    rbf_svr1= function1()  
    dataDict1 = test_x_1.to_dict()
    predictResult1=rbf_svr1.predict(test_x_1) 
    dataDict1['Predicted S2'] = pd.Series(predictResult1,index=list(range(2, 19269))).round(5)
    data1 = pd.DataFrame(dataDict1)
    data1.to_excel(r'Predicted_S2_BRR.xlsx',sheet_name='Predicted',index=True,index_label="index")