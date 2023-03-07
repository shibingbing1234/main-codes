# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:08:16 2022

@author: Tianjiao Liu
"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Reshape,Dropout,Activation
from tensorflow.keras.layers import Conv1D,AveragePooling1D
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from math import*

#Define a function in the way of def, function () 1 is the function name, the parameters can be passed in parentheses, I do not pass here
def function1(): #Calculate the function of the first paragraph
##2.input data
    data= pd.read_csv('S1 Input train data.csv') # well data
    AC = data.values[:832, 2:3]
    SP = data.values[:832, 3:4]

    LLD = data.values[:832, 5:6]
    LLS = data.values[:832, 6:7] 

#Training feature data
    X=np.concatenate((AC,SP,LLD,LLS),axis=1)
#Training target data
    S2 = data.values[:832, 1:2]
    X_train, X_test, y_train, y_test = train_test_split(X, S2, test_size=0.3,random_state=5)     
    model1 = Sequential()
    model1.add(Conv1D(4, 2,input_shape=(4,1),padding='same',activation='PReLU',))
    model1.add(Conv1D(8, 2,activation='PReLU'))
    model1.add(Conv1D(12, 2,activation='PReLU'))
    model1.add(Conv1D(16, 2,activation='PReLU'))
    model1.add(AveragePooling1D(pool_size=2,padding='same'))  #average pooling
    model1.add(Flatten())
    model1.add(Dense(1,activation='relu'))
    learning_rate=0.01
    adam=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    eba=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ebk=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=20, factor = 0.5, min_lr=0.00001,restore_best_weights=True)
    model1.compile(optimizer=adam, loss='mse')
    history1=model1.fit(X_train,y_train,epochs=1000,batch_size=12,verbose=2,callbacks=[eba,ebk],validation_data= (X_test, y_test))
    predictA1 = model1.predict(X_test)
    print('r2_score: %.2f' % r2_score(y_test, predictA1))
    predictA2= model1.predict(X)
    print('r2_score: %.2f' % r2_score(S2, predictA2))

    plt.figure('Prediction&TrueValues---Error',dpi=300)
    plt.grid(b=None, which='both', axis='both' )
    plt.plot(history1.history['loss'])  
    plt.plot(history1.history['val_loss'])  
    plt.title('Model loss')  
    plt.yscale("log")
    plt.xlim((0,65))
    plt.ylabel('Mean Squnared Error(mse)')  
    plt.xlabel('Lterations"45Epochs"')  
    plt.vlines([45], 1,8, linestyles='dashed', colors='grey')
    plt.legend(['Train', 'Test'], loc='upper left')  
    plt.show()


    predictA1 = model1.predict(X_test)
    plt.figure('Prediction&TrueValues---Error',dpi=300)
    plt.grid(b=None, which='major', axis='both' )
    plt.scatter(y_test,predictA1,marker='x',color='red')
    x=np.linspace(1,8.0,100)
    z=x
    plt.xlabel('Actual TOC(wt%)')
    plt.ylabel('predicted TOC(wt%)')
    plt.xlim((1,8))
    plt.ylim((1,8))
    plt.plot(x,z,color='black',linewidth=1.0)
    plt.text(0.3, 1.50, 'R$\mathregular{^2}$=0.22',fontdict={'weight': 'normal', 'size': 15})
    plt.show()
    print('r2_score: %.2f' % r2_score(y_test, predictA1))

    predictA2= model1.predict(X)
    plt.figure('Prediction&TrueValues---Error',dpi=300)
    plt.grid(b=None, which='major', axis='both' )
    plt.scatter(S2,predictA2,marker='x',color='red')
    x=np.linspace(1,8.0,100)
    z=x
    plt.xlim((1,8))
    plt.ylim((1,8))
    plt.xlabel('Actual S2')
    plt.ylabel('predicted S2')
    plt.plot(x,z,color='black',linewidth=1.0,)
    plt.text(0.3, 1.50, 'R$\mathregular{^2}$=0.28',fontdict={'weight': 'normal', 'size': 15})
    plt.show()
    print('r2_score: %.2f' % r2_score(S2, predictA2))
    return model1  


    
if __name__ == '__main__':
    #915~1158.3m   This is the prediction data to pass in the model
    datasetFilename_1 = 'S2 Input dataset for predict S2.CSV'
    test_x_1 = pd.read_csv(datasetFilename_1, usecols=['AC','SP','LLD','LLS'])
    test_x_1=test_x_1.loc[2:19270,['AC','SP','LLD','LLS']]
    #Call the first function, return the rbf_svr1 model, and pass in the data you want to predict
    model1= function1()  
    dataDict1 = test_x_1.to_dict()
    predictResult1=model1.predict(test_x_1)
    print(predictResult1)
    data1 = pd.DataFrame(predictResult1)
    data1.to_excel(r'Predicted_S2_all_CNN.xlsx',sheet_name='Predicted',index=True,index_label="index")

 