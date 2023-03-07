# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:56:55 2022

@author: Tianjiao Liu
"""
import numpy as np
import xgboost as xgb

import pandas as pd
def function1():  #
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
        'max_depth':[3,4,5,6,7,8],          # Maximum depth per tree, default 6；
        'learning_rate':[0.2,0.3,0.4,0.5,0.6],      # Learning rate, which is multiplied by the predicted result of each tree, defaults to 0.3；
        'n_estimators':[50,100,150,200],        # How many trees to fit can also be understood as how many iterations. Default 100；
        'silent':['True','False'],
        'objective':['reg:linear','reg:squarederror'],   # This default parameter is different from XGBClassifier
        'booster':['gbtree','glinear'],         #There are two models to choose gbtree and gblinear. Gbtree uses tree-based model for lifting calculation, and gblinear uses linear model for lifting calculation. default to gbtree
        'gamma':[0],                 # Minimum ' loss reduction ' required for further splitting on leaf nodes. Default 0；
        'min_child_weight':[1,2,3,4],      # Can be understood as the minimum number of samples of leaf nodes, default 1；
        'subsample':[0.1,0.3,0.5,0.7,0.9,1],              # The training set sampling proportion, each time before fitting a tree, will carry out the sampling step. Default 1, value range ( 0, 1 ]
        'colsample_bytree':[0.7,0.8,0.9,1],       # Before fitting a tree each time, we decide how many features to use, parameter default 1, value range ( 0,1 ].
        'reg_alpha':[0],             #The default value is 0, which controls the L1 regularization parameter of the weight value of the complexity of the model. The larger the parameter value, the more difficult the model is to overfit.
        'reg_lambda':[1],            # The default is 1, and the L2 regularization parameter of the weight value of the model complexity is controlled. The larger the parameter, the less likely the model is to overfit.             
        'random_state':[0] # random seed
    } 
  
    #Training with the XGBoost Regressor ( ) model
    rfc=xgb.XGBRegressor() 
    #Optimize models with grid search and 5-fold cross-validation to find optimal parameters
    rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    rfc_cv.fit(X_train, S2)
    #Get the best model
    best_model1 = rfc_cv.best_estimator_
    #Parameters and Kernel Functions of the Best Model for Print Output
    print(rfc_cv.best_params_)
   
    return best_model1    #This is the model to calculate the second return.



if __name__ == '__main__':
    #915 ~ 1158.3m This is the predcted data to be passed into the model
    datasetFilename_1 = 'S2 Input dataset for predict S2.CSV'
    test_x_1 = pd.read_csv(datasetFilename_1, usecols=['AC','SP','LLD','LLS'])
    test_x_1=test_x_1.loc[2:19270,['AC','SP','LLD','LLS']]
    #Call the function, return the model, and then pass in the data to be predicted
    rbf_svr1= function1()  
    dataDict1 = test_x_1.to_dict()
    predictResult1=rbf_svr1.predict(test_x_1) 
    dataDict1['Predicted S2'] = pd.Series(predictResult1,index=list(range(2, 19269))).round(5)
    data1 = pd.DataFrame(dataDict1)
    data1.to_excel(r'Predicted_S2_all_XGboost.xlsx',sheet_name='Predicted',index=True,index_label="index")