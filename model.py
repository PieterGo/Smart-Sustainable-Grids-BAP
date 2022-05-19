# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import pandas, numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

def readInputFile(filename):
    
    # This method loads all the data from the .xlsx file. You do not need to modify anything here.
    
    LoadData = pandas.read_excel(filename, sheet_name= 'Load', index_col=0) #data of load entire neighbourhood
    WeatherData = pandas.read_excel(filename, sheet_name= 'Weather', index_col=0) #data of the weather
    TransformerData = pandas.read_excel(filename, sheet_name= 'Transformer', index_col=0) #rated power transformer
    PVSystem = pandas.read_excel(filename, sheet_name='PVSystem', index_col=0) 
    StorageData = pandas.read_excel(filename, sheet_name= 'StorageSystem', index_col=0) # batterij
    EVDemand= pandas.read_excel(filename, sheet_name = 'EVDemand', index_col=0) #EV charging data


    # We could return multiple items but it is preferable to return a structure, in this case a dictionary 
    return {'WeatherData':WeatherData, 'PVSystem':PVSystem, 'StorageData':StorageData,
            'LoadData':LoadData, 'TransformerData':TransformerData, 'EVDemand':EVDemand}

filename = 'variables_BAP.xlsx'
data = readInputFile(filename)
print(data)





