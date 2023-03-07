
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:06:44 2022

@author: Tianjiao Liu
"""
##1.import databases
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn import metrics
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import time
from scipy import stats
import xgboost as xgb

##2.import data
data= pd.read_csv('S1 Input train data.csv') # logging data
AC = data.values[:832, 2:3]
SP = data.values[:832, 3:4]

LLD = data.values[:832, 5:6]
LLS = data.values[:832, 6:7] 

#Training feature data
traffic_feature=np.concatenate((AC,SP,LLD,LLS),axis=1)
#Training target data
traffic_target = data.values[:832, 1:2]


feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, train_size=0.995,random_state=17)

##4.Test the effect of single XGBoost
start1=time.time()
model_svr = xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =0.01, n_estimators=50, max_depth=3,
                                         min_child_weight=1, subsample=0.1, colsample_bytree=0.5) 
model_svr.fit(feature_train,target_train)
predict_results1=model_svr.predict(feature_test)
end1=time.time()
 
SVRRESULT=predict_results1
plt.plot(target_test)#Test array
plt.plot(predict_results1)#Test array
plt.legend(['True','XGBoost'])
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.title("XGBoost")  # heading
print("EVS:",explained_variance_score(target_test,predict_results1))
print("R2:",metrics.r2_score(target_test,predict_results1))
print("Time:",end1-start1)


##6.t-SNE dimensionality reduction
from sklearn.manifold import TSNE
newData = TSNE(n_components=3,random_state=17, learning_rate=100, init='pca').fit_transform(traffic_feature)
plt.scatter(newData[:, 0],newData[:, 1],newData[:, 2])
plt.title("PCA-TSNE")  # heading
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()

##7.Import kmeans and DBSCAN for noise reduction comparison
#Kmeans
t0 = time.time()
kmeans = KMeans(init = 'k-means++',n_clusters=10, random_state=10).fit(newData)
t = time.time() - t0
plt.scatter(newData[:, 0], newData[:, 1],newData[:, 2],c=kmeans.labels_)
plt.title('Kmeans time : %f'%t)
fig = plt.gcf() 
fig.set_size_inches(18.5, 10.5)
plt.show()

#DBSCAN
# Build an empty list to hold results under different parameter combinations
res = []
# Iteration of different eps values
for eps in np.arange(0.001,1,0.005):
    # Iterating different min_samples values
    for min_samples in range(2,12):
        t0 = time.time()
        dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(newData)
        t = time.time()-t0
        plt.scatter(newData[:, 0], newData[:, 1],newData[:, 2],c=dbscan.labels_)
        plt.title('DBSCAN time: %f'%t)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.show()
        # Count the number of clusters under each parameter combination ( -1 represents outliers )
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        # Number of Outliers
        outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
        # Count the number of samples per cluster
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats})
# Store the iteration results in a data box     
df = pd.DataFrame(res)

# Screening reasonable parameter combinations according to conditions
df.loc[df.n_clusters == 3, :]


#DBSCAN Noise Reduction
dbscan = DBSCAN(eps=0.996,min_samples=11).fit(newData)
 
YYY=[]
XXX=[]
C=[]
for inx,i in enumerate(dbscan.labels_):
    if i==-1:
        YYY.append(i)
        XXX.append(newData[inx])
        C.append(inx)
        
XXX=np.array(XXX)
XXX=XXX.astype(np.float64)
plt.scatter(XXX[:, 0], XXX[:, 1], XXX[:, 2],c=YYY)
fig = plt.gcf()
fig.set_size_inches(9, 9)
plt.title('DBSCAN After noise reduction time: %f'%t)
plt.show()

##Data restoration
XXX
CCC=[]
DDD=[]
a=1
 
for inx,i in enumerate(traffic_target):
    for j in C:
        if(inx==j):
            a=1+a
            CCC.append(i)
            DDD.append([a,i])
        
 
CCC=np.array(CCC)
DDD=np.array(DDD)
 
DDD.shape
dbscan = DBSCAN(eps=0.996,min_samples=11).fit(DDD)
plt.scatter(DDD[:,0],DDD[:,1],c=dbscan.labels_)
plt.title('DBSCAN')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()

##9.XGBoost prediction of restored data
after_feature_train,after_feature_test,after_target_train, after_target_test = train_test_split(XXX,CCC,train_size=0.995,random_state=17)
start=time.time()
model_svr = xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =0.1, n_estimators=10, max_depth=2,
                                         min_child_weight=1, subsample=0.1, colsample_bytree=0.1) 
model_svr.fit(XXX,CCC)
after_predict_result=model_svr.predict(after_feature_test)
end=time.time()
 
DBSCANSVRRESULT=after_predict_result
plt.plot(after_predict_result)#Test array
plt.plot(after_target_test)#Test array
plt.title("DBSCAN-XGBoost")  # Heading
plt.legend(['True','DBSCAN-XGBoost'])
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
print("EVS:",explained_variance_score(after_target_test,after_predict_result))
print("R2:",metrics.r2_score(after_target_test,after_predict_result))
print("Time:",end-start)

##10.adding PSO
class PSO:
    def __init__(self, parameters):
        """
        particle swarm optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # initialization
        self.NGEN = parameters[0]    # Algebra of iteration
        self.pop_size = parameters[1]    # population size
        self.var_num = len(parameters[2])     # a variable quantity
        self.bound = []                 # Constraint range of variables
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
 
        self.pop_x = np.zeros((self.pop_size, self.var_num))    # Position of all particles
        self.pop_v = np.zeros((self.pop_size, self.var_num))    # Velocity of all particles
        self.p_best = np.zeros((self.pop_size, self.var_num))   # The optimal position of each particle
        self.g_best = np.zeros((1, self.var_num))   # Globally optimal location
 
        # Initialize the 0th generation initial global optimal solution
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]      # Store the best individuals
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit
 
    def fitness(self, ind_var):

        """
        Individual fitness calculation
        """
        X = XXX
        y = CCC
        RRR=after_feature_test.tolist()
        LLL=after_target_test.tolist()
        c = float(ind_var[0])
        e = int(ind_var[1])
        g = int(ind_var[2])
        m = int(ind_var[3])
        s = float(ind_var[4])
        o = float(ind_var[5])
        print("c:",c)
        print("e:",e)
        print("g:",g)
        print("m:",m)
        print("s:",s)
        print("o:",o)        
        if c==0:c=0.001
        if e==0:e=1
        if g==0:g=1
        if m==0:m=1
        if s==0:s=0.001
        if o==0:o=0.001
        c = float(c)
        e = int(e)
        g = int(g)
        m = int(m)
        s = float(s)
        o = float(o)            
        clf = xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =c, n_estimators=e, max_depth=g,
                                                 min_child_weight=m, subsample=s, colsample_bytree=o)
        clf.fit(X, y)
        predictval=clf.predict(RRR)
        print("R2 = ",metrics.r2_score(LLL,predictval)) # R2
        return  metrics.r2_score(LLL,predictval)
 
    def update_operator(self, pop_size):
        """
        Update operator : Updates the position and velocity at the next moment
        """
        c1 = 2     # Learning factor, generally 2
        c2 = 2
        w_start = 0.9;  # initial inertia weight
        w_end = 0.3; # The inertia weight of particle swarm at the maximum number of iterations
        for i in range(pop_size):
            w = w_start - (w_start - w_end) * i / NGEN;  # Update inertia weight
            # renewal speed
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                        self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
            # types of regeneration site
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # Cross-border protection
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # Update p _ best and g _ best
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
 
    def main(self):
        popobj = []
        self.ng_best = np.zeros((1, self.var_num))[0]
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()
            print('Best location：{}'.format(self.ng_best))
            print('Maximum function value：{}'.format(self.fitness(self.ng_best)))
        print("---- End of (successful) Searching ----")
 
        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()
 
    def printResult(self):
        '''
        plot the result of the genetic algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        y3 = self.trace[:, 2]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.plot(x, y3, 'b', label='max value')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.xlabel("GENS")
        plt.ylabel("R2")
        plt.title("GA")
        plt.legend()
        plt.show()
        
if __name__ == '__main__':
    NGEN = 150
    popsize = 40
    low = [0.01,50,3,1,0.1,0.5]
    up = [0.4,200,8,6,1,1]
    parameters = [NGEN, popsize, low, up]
    pso = PSO(parameters)
    pso.main()
   
##11.Substitute the optimized parameters into XGBoost
after_feature_train,after_feature_test,after_target_train, after_target_test = train_test_split(XXX,CCC,train_size=0.995,random_state=17)
svr = xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =0.4, n_estimators=200, max_depth=8,
                                         min_child_weight=2, subsample=0.64421357, colsample_bytree=1.) 
start=time.time()
svr.fit(XXX,CCC)
after_predict_result=svr.predict(after_feature_test)
end=time.time()
DBSCANSVRGARESULT=[]
DBSCANSVRGARESULT=after_predict_result
plt.plot(after_predict_result,marker='o')#Test array
plt.plot(after_target_test,marker='x')
 
plt.title("DBSCAN-XGBoost-GA")  # Heading
plt.legend(['True','DBSCAN-XGBoost-GA'])
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
print("EVS:",explained_variance_score(after_target_test,after_predict_result))
print("R2:",metrics.r2_score(after_target_test,after_predict_result))
print("Time:",end-start)

##12.Comparison with GA-XGBoost
start1=time.time()
model_svr =xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =0.4, n_estimators=200, max_depth=8,
                                         min_child_weight=2, subsample=0.64421357, colsample_bytree=1.) 
model_svr.fit(feature_train,target_train)
predict_results1=model_svr.predict(feature_test)
SVRGARESULT=[]
SVRGARESULT=predict_results1
end1=time.time()
plt.plot(target_test)#Test array
plt.plot(predict_results1)#Test array
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.title("XGBoost-GA")  # Heading
plt.legend(['True','XGBoost-GA'])
plt.show()
print("EVS:",explained_variance_score(target_test,predict_results1))
print("R2:",metrics.r2_score(target_test,predict_results1))
print("Time:",end1-start1)


##13.For prediction
datasetFilename_1 = 'S2 Input dataset for predict S2.CSV'
test_x_1 = pd.read_csv(datasetFilename_1, usecols=['AC','SP','LLD','LLS'])
test_x_1=test_x_1.loc[2:19270,['AC','SP','LLD','LLS']]
  #Call the function, return the model, and then pass in the data to be predicted
dataDict1 = test_x_1.to_dict()
rbf_svr = xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =0.4, n_estimators=200, max_depth=8,
                                         min_child_weight=2, subsample=0.64421357, colsample_bytree=1.)  
rbf_svr.fit(feature_train,target_train)
predictResult1=rbf_svr.predict(test_x_1)
dataDict1['Predicted S2'] = pd.Series(predictResult1,index=list(range(2, 19269))).round(5)
data1 = pd.DataFrame(dataDict1)
data1.to_excel(r'Predicted_S2_all_t-SNE+DBSCAN+PSO_XGBoost.xlsx',sheet_name='Predicted',index=True,index_label="index")