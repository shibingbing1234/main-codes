The code for S2 prediction. The code includes the traditional regression methods and our proposed improved method.

1、A set of ML regression algorithms including support vector regression (SVR), random forest (RF), ElasticNet, extreme gradient boosting (XGBoost), convolutional neural network (CNN), and Bayesian ridge regression (BRR) were compared to select the most suitable algorithm for predicting S2.

2、Data preprocessing, that is, normalisation, moving average filtering homogenisation, t-distributed stochastic neighbour embedding (t-SNE) dimension reduction, and density-based spatial clustering of application with noise (DBSCAN) reduction, were carried out to eliminate the error between the abnormal well-logging data and the corresponding depth of the measured points.

3、The violin plot and Taylor diagram were also applied to compare the reliability and effectiveness of the RF, SVR, XGBoost, ElasticNet, BRR and CNN models.

4、Among the available ML regression algorithms, the XGBoost algorithm is an excellent tool for the S2 evaluation.

5、From the optimisation algorithms, the particle swarm optimisation (PSO) and genetic algorithm (GA) with good optimisation effects were selected to optimise the hyperparameters of the XGBoost regression model.

6、the improved XGBoost (t-SNE+ DBSCAN+ PSO _XGBoost) algorithm optimised by PSO can effectively predict S2 in source rock, becoming a powerful tool for source rock or shale oil resource evaluation.
