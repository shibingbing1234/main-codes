# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 19:17:38 2022

@author: Tianjiao Liu
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas  as pd
import skill_metrics as sm
 
data = pd.read_excel(r"Taylor_S2.xlsx")
# Calculate statistics for Taylor diagram
# The first array element (e.g. taylor_stats1[0]) corresponds to the 
# reference series while the second and subsequent elements
# (e.g. taylor_stats1[1:]) are those for the predicted series.
taylor_stats1 = sm.taylor_statistics(data.XGBoost,data.ref,'data')
taylor_stats2 = sm.taylor_statistics(data.PSO_XGBoost,data.ref,'data')
taylor_stats3 = sm.taylor_statistics(data.GA_XGBoost,data.ref,'data')
taylor_stats4 = sm.taylor_statistics(data.ElasticNet,data.ref,'data')
taylor_stats5 = sm.taylor_statistics(data.BRR,data.ref,'data')
taylor_stats6 = sm.taylor_statistics(data.RF,data.ref,'data')
taylor_stats7 = sm.taylor_statistics(data.SVR,data.ref,'data')
taylor_stats8 = sm.taylor_statistics(data.CNN,data.ref,'data')
# Store statistics in arrays
sdev = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1], 
                 taylor_stats2['sdev'][1], taylor_stats3['sdev'][1],taylor_stats4['sdev'][1],
                 taylor_stats5['sdev'][1], taylor_stats6['sdev'][1],taylor_stats7['sdev'][1],taylor_stats8['sdev'][1]])
crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1], 
                  taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1], taylor_stats4['crmsd'][1],
                  taylor_stats5['crmsd'][1], taylor_stats6['crmsd'][1], taylor_stats7['crmsd'][1], taylor_stats8['crmsd'][1]])
ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1],   
                  taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1], taylor_stats4['ccoef'][1],
                  taylor_stats5['ccoef'][1], taylor_stats6['ccoef'][1], taylor_stats7['ccoef'][1], taylor_stats8['ccoef'][1]])
'''
    Produce the Taylor diagram

    Note that the first index corresponds to the reference series for 
    the diagram. For example sdev[0] is the standard deviation of the 
    reference series and sdev[1:4] are the standard deviations of the 
    other 3 series. The value of sdev[0] is used to define the origin 
    of the RMSD contours. The other values are used to plot the points 
    (total of 3) that appear in the diagram.

    For an exhaustive list of options to customize your diagram, 
    please call the function at a Python command line:
    >> taylor_diagram
''' 
# Set the basic configuration of matplotlib
rcParams["figure.figsize"] = [6, 6]
rcParams["figure.facecolor"] = "white"
rcParams["figure.edgecolor"] = "white"
rcParams["figure.dpi"] = 160
rcParams['lines.linewidth'] = 1 # 
rcParams["font.family"] = "Times New Roman"
rcParams.update({'font.size': 12}) # 
plt.close('all')
# Start drawing
sm.taylor_diagram(sdev,crmsd,ccoef,
                      markerDisplayed = 'colorBar', titleColorbar = 'RMSD',
                     locationColorBar = 'EastOutside',
                      cmapzdata = crmsd, titleRMS = 'off',
                      tickRMS = range(0,3,1), tickRMSangle = 110.0,
                      colRMS = 'm', styleRMS = ':', widthRMS = 2.0,
                      tickSTD = range(0,0.5,1), axismax = 1.0,
                      colSTD = 'k', styleSTD = '-', widthSTD = 1.5,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0)
text_font = {'size':'15','weight':'bold','color':'black'}
plt.title("Taylor_diagram of different models in Python",fontdict=text_font,pad=35)

data1 = pd.DataFrame(sdev)
data1.to_excel(r'sdev.xlsx',sheet_name='Predicted',index=True,index_label="index")
data2 = pd.DataFrame(crmsd)
data2.to_excel(r'crmsd.xlsx',sheet_name='Predicted',index=True,index_label="index")
data3 = pd.DataFrame(ccoef)
data3.to_excel(r'ccoef.xlsx',sheet_name='Predicted',index=True,index_label="index")
data4 = pd.DataFrame(taylor_stats4)
data4.to_excel(r'taylor_stats4.xlsx',sheet_name='Predicted',index=True,index_label="index")