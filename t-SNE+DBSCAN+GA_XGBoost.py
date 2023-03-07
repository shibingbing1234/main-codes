# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:48:23 2022

@author: Tianjiao Liu
"""
##1.Introducing databases

from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd 
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats
import xgboost as xgb
from sklearn import metrics
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
plt.show()
print("EVS:",explained_variance_score(target_test,predict_results1))
print("R2:",metrics.r2_score(target_test,predict_results1))
print("Time:",end1-start1)



##6.t-SNE reduction dimension
from sklearn.manifold import TSNE
newData = TSNE(n_components=3,random_state=17, learning_rate=100, init='pca').fit_transform(traffic_feature)
fig=plt.scatter(newData[:, 0],newData[:, 1],newData[:, 2])
plt.title("PCA-TSNE")  # heading
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()

##7.Import kmeans and DBSCAN for noise reduction comparison
#Kmeans
t0 = time.time()
kmeans = KMeans(init = 'k-means++',n_clusters=10, random_state=10).fit(newData)
t = time.time() - t0
plt.scatter(newData[:, 0], newData[:, 1],newData[:, 2],c=kmeans.labels_)
plt.title('Kmeans time : %f'%t)
fig = plt.gcf() 
fig.set_size_inches(8, 6)
plt.show()

#DBSCAN
# Build an empty list to hold results under different parameter combinations
res = []
# Iteration of different eps values
for eps in np.arange(0.001,1,0.005):
    # Iterating different min _ samples values
    for min_samples in range(2,12):
        t0 = time.time()
        dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(newData)
        t = time.time()-t0
        plt.scatter(newData[:, 0], newData[:, 1],newData[:, 2],c=dbscan.labels_)
        plt.title('DBSCAN time: %f'%t)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
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


#DBSCAN after noise reduction
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
fig.set_size_inches(8, 6)
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
plt.title("DBSCAN-XGBoost")  # heading
plt.legend(['True','DBSCAN-XGBoost'])
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
print("EVS:",explained_variance_score(after_target_test,after_predict_result))
print("R2:",metrics.r2_score(after_target_test,after_predict_result))
print("Time:",end-start)

##10.adding GA
XXX = XXX.tolist()
CCC = CCC.tolist()
RRR=after_feature_test.tolist()
LLL=after_target_test.tolist()
def msefunc(predictval,realval):
   squaredError = []
   absError = []
   for i in range(len(predictval)):
       val=predictval[i-1]-realval[i-1]
       squaredError.append(val * val)  # target-prediction difference squared
 
   print("R2 = ",metrics.r2_score(realval,predictval)) # R2
   return metrics.r2_score(realval,predictval)
 
def SVMResult(vardim, x, bound):
    X = XXX
    y = CCC
    c=float(x[0])
    e=int(x[1])
    g=int(x[2])
    m=int(x[3])
    s=float(x[4])
    o=float(x[5])
    print("c:",c)
    print("e:",e)
    print("g:",g)
    print("m:",m)
    print("s:",s)
    print("o:",o)
    clf = xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =c, n_estimators=e, max_depth=g,
                                             min_child_weight=m, subsample=s, colsample_bytree=o) 
    print("start")
    clf.fit(X, y)
    print("finish")
    predictval=clf.predict(RRR)
    return msefunc(predictval,LLL)
class GAIndividual:
 
    '''
    individual of genetic algorithm
    '''
 
    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
 
    def generate(self):
        '''
        generate a random chromsome for genetic algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]
 
    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = SVMResult(self.vardim, self.chrom, self.bound)
import random
import copy
 
 
class GeneticAlgorithm:
 
    '''
    The class for genetic algorithm
    '''
 
    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        param: algorithm required parameters, it is a list which is consisting of crossover rate, mutation rate, alpha
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 3))
        self.params = params
 
    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)
 
    def evaluate(self):
        '''
        evaluation of the population fitnesses
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness
 
    def solve(self):
        '''
        evolution process of genetic algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluate()
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.maxfitness = np.max(self.fitness)
        
        self.trace[self.t, 0] =  self.best.fitness
        self.trace[self.t, 1] =  self.avefitness
        self.trace[self.t, 2] =  self.maxfitness
        print("Generation %d: optimal function value is: %f; average function value is %f;max function value is %f"% (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1],self.trace[self.t, 2]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.selectionOperation()
            self.crossoverOperation()
            self.mutationOperation()
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.maxfitness = np.max(self.fitness)
            
            self.trace[self.t, 0] =  self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            self.trace[self.t, 2] =  self.maxfitness
            print("Generation %d: optimal function value is: %f; average function value is %f;max function value is %f"% (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1],self.trace[self.t, 2]))
 
        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print ("Optimal solution is:")
        print (self.best.chrom)
        self.printResult()
 
    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        '''
        newpop = []
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.sizepop, 1))
 
        sum1 = 0.
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness
            sum1 = accuFitness[i]
 
        for i in range(0, self.sizepop):
            r = random.random()
            idx = 0
            for j in range(0, self.sizepop - 1):
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        self.population = newpop
 
    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()
            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):
                    newpop[i].chrom[j] = newpop[i].chrom[
                        j] * self.params[2] + (1 - self.params[2]) * newpop[i + 1].chrom[j]
                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                        (1 - self.params[2]) * newpop[i].chrom[j]
        self.population = newpop
 
    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] - (newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * (1 - random.random() ** (1 - self.t / self.MAXGEN))
                else:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] + (self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * (1 - random.random() ** (1 - self.t / self.MAXGEN))
        self.population = newpop
 
    def printResult(self):
        '''
        plot the result of the genetic algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        y3 = self.trace[:, 2]
        plt.plot(x, y1, 'r', label='optimal value')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.xlabel("GENS")
        plt.ylabel("R2")
        plt.title("GA")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
 
   bound = (np.array([[0.01,50,3,1,0.1,0.5],[0.4,200,8,6,1,1]]))
    
   ga = GeneticAlgorithm(19, 6, bound, 300, [0.75, 0.25, 0.5])
   ga.solve()
   
##11.Substitute optimized parameters into XGBoost
after_feature_train,after_feature_test,after_target_train, after_target_test = train_test_split(XXX,CCC,train_size=0.995,random_state=17)
svr = xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =2.55732567e-02, n_estimators=180, max_depth=5,
                                         min_child_weight=5, subsample=7.84408703e-01, colsample_bytree=7.77215195e-01) 
start=time.time()
svr.fit(XXX,CCC)
after_predict_result=svr.predict(after_feature_test)
end=time.time()
DBSCANSVRGARESULT=[]
DBSCANSVRGARESULT=after_predict_result
plt.plot(after_predict_result,marker='o')#Test array
plt.plot(after_target_test,marker='x')
 
plt.title("DBSCAN-XGBoost-GA")  # heading
plt.legend(['True','DBSCAN-XGBoost-GA'])
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
print("EVS:",explained_variance_score(after_target_test,after_predict_result))
print("R2:",metrics.r2_score(after_target_test,after_predict_result))
print("Time:",end-start)

##12.Comparison with GA-XGBoost
start1=time.time()
model_svr =xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =2.55732567e-02, n_estimators=180, max_depth=5,
                                         min_child_weight=5, subsample=7.84408703e-01, colsample_bytree=7.77215195e-01) 
model_svr.fit(feature_train,target_train)
predict_results1=model_svr.predict(feature_test)
SVRGARESULT=[]
SVRGARESULT=predict_results1
end1=time.time()
plt.plot(target_test)#Test array
plt.plot(predict_results1)#Test array
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.title("XGBoost-GA")  # heading
plt.legend(['True','XGBoost-GA'])
plt.show()
print("EVS:",explained_variance_score(target_test,predict_results1))
print("R2:",metrics.r2_score(target_test,predict_results1))
print("Time:",end1-start1)


##13.For prediction
datasetFilename_1 = 'S2 Input dataset for predict S2.CSV'
test_x_1 = pd.read_csv(datasetFilename_1, usecols=['AC','SP','LLD','LLS'])
test_x_1=test_x_1.loc[2:19270,['AC','SP','LLD','LLS']]
  #Call the first function, return the model, and pass in the data you want to predict
dataDict1 = test_x_1.to_dict()
rbf_svr = xgb.XGBRegressor(booster='gbtree',gamma=0,objective='reg:squarederror',random_state=0,reg_alpha=0,reg_lambda=1,learning_rate =2.55732567e-02, n_estimators=180, max_depth=5,
                                         min_child_weight=5, subsample=7.84408703e-01, colsample_bytree=7.77215195e-01)  
rbf_svr.fit(feature_train,target_train)
predictResult1=rbf_svr.predict(test_x_1)
dataDict1['Predicted S2'] = pd.Series(predictResult1,index=list(range(2, 19269))).round(5)
data1 = pd.DataFrame(dataDict1)
data1.to_excel(r'Predicted_S2_all_t-SNE+DBSCAN+GA_XGBoost.xlsx',sheet_name='Predicted',index=True,index_label="index")