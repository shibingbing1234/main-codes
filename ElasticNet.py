# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:49:25 2022

@author: Tianjiao Liu
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV
import pandas as pd
def function1():  #Calculate the function of the second section
    from sklearn.model_selection import GridSearchCV 
    data= pd.read_csv('S1 Input train data.csv') # logging data
    AC = data.values[:832, 2:3]
    SP = data.values[:832, 3:4]

    LLD = data.values[:832, 5:6]
    LLS = data.values[:832, 6:7] 

#Training target data
    S2 = data.values[:832, 1:2]
    X_train=np.concatenate((AC,SP,LLD,LLS),axis=1)
    #Here is the parameter range for the max_depth/n_estimators/max_features/min_samples_split that the model requires
    param_grid = {
        'l1_ratio':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
        'fit_intercept':['False'],
        'max_iter':[10000000], 
        'tol':[0.0001], 
        'selection':['random']
    } 
  
    #Elasticnet Regressor ( ) model training
    rfc=ElasticNetCV() 
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
    data1.to_excel(r'Predicted_S2_all_EN.xlsx',sheet_name='Predicted',index=True,index_label="index")