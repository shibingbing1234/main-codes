# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:18:01 2022

@author: Tianjiao Liu
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datasetFilename = 'S3 Correlation data between well logging data and S2.CSV'
mydata = pd.read_csv('S3 Correlation data between well logging data and S2.CSV', usecols=('S2', 'GR', 'DEN', 'LLS', 'SP', 'AC', 'LLD'))

print(mydata.corr()) #Calling the corr method by the data frame will calculate the similarity between each column, and the return is a rectangle
Data=mydata.corr()
fig, ax = plt.subplots(figsize = (26,26))
#If you want to add the ticklabels of the horizontal axis and the number axis to the heat map of a two-dimensional array, you can either directly generate it by converting the array into a DataFrame with column and index, or you can add it later. If added later, it is more flexible, including setting the size and direction of the labels.
sns.heatmap(Data, 
                annot=True, vmax=1,vmin = -1, xticklabels= True, yticklabels= True, square=True, cmap="rainbow")
#sns.heatmap(annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
#            square=True, cmap="YlGnBu")
ax.set_title('Unconsolidated section Heatmap', fontsize = 30)
ax.set_ylabel('Y', fontsize = 26)
ax.set_xlabel('X', fontsize = 26) #It is the same as the original layout of the matrix
ax.set_yticklabels(['S2','GR', 'DEN', 'LLS', 'SP', 'AC', 'LLD'], fontsize = 26, rotation = 360, horizontalalignment='right')
ax.set_xticklabels(['S2', 'GR', 'DEN', 'LLS', 'SP', 'AC', 'LLD'], fontsize = 26, horizontalalignment='right')

