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
    
    WeatherData = pandas.read_excel(filename, sheet_name= 'Weather', index_col=0) #data of the weather
    # SystemDemand= pandas.read_excel(filename, sheet_name = 'SystemDemand', index_col= 0) #user data
    # EVDemand= pandas.read_excel(filename, sheet_name = 'EVDemand', index_col= 0) #EV charging data
    PVSystem = pandas.read_excel(filename, sheet_name='PVSystem', index_col=0) 
    # LoadData = pandas.read_excel(filename, sheet_name = 'Loads', index_col= 0) # wat moeten we hiermee ivm groep A? @nanda
    # PVData = pandas.read_excel(filename, sheet_name='PVParks', index_col=0) # wat moeten we hiermee ivm groep A? @nanda
    StorageData = pandas.read_excel(filename, sheet_name= 'StorageSystem', index_col=0) # batterij
    # UnitData = pandas.read_excel(filename, sheet_name = 'Generators', index_col= 0) # transformer data (bij ons alleen min&max)


#  'EVDemand':EVDemand,    

    # We could return multiple items but it is preferable to return a structure, in this case a dictionary 
    return {'WeatherData':WeatherData, 'PVSystem':PVSystem, 'StorageData':StorageData}
            #'SystemDemand':SystemDemand, 'LoadData':LoadData, 'PVData':PVData, 'UnitData':UnitData}

filename = 'variables_BAP.xlsx'
data = readInputFile(filename)
print(data)





