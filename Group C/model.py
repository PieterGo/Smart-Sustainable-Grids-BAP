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
    
    # Load all data from excel sheets
    
    TimeData = pandas.read_excel(filename, sheet_name='Time', index_col=0) #time data including time step
    LoadData = pandas.read_excel(filename, sheet_name= 'Load', index_col=0) #data of load entire neighbourhood
    TransformerData = pandas.read_excel(filename, sheet_name= 'Transformer', index_col=0) #rated power transformer
    PVProduction = pandas.read_excel(filename, sheet_name='PVProduction', index_col=0) # omgerekende irradiation
    StorageData = pandas.read_excel(filename, sheet_name= 'StorageSystem', index_col=0) # batterij
    EVDemand= pandas.read_excel(filename, sheet_name = 'EVDemand', index_col=0) #EV charging data

    # Return directory 
    return {'PVProduction':PVProduction, 'StorageData':StorageData, 'TimeData':TimeData,
            'LoadData':LoadData, 'TransformerData':TransformerData, 'EVDemand':EVDemand}

def optimizationModel(inputData, modelType):   
    
    # Unpack the data from the dictionary
    LoadData = inputData['LoadData']
    TransformerData = inputData['TransformerData']
    PVProduction = inputData['PVProduction']
    StorageData = inputData['StorageData']
    EVDemand = inputData['EVDemand']
    TimeData = inputData['TimeData']
#-------------------------------------------------------------------
    # Define the Model
    model = ConcreteModel()

#------------------------------------------------------------------
    #Define Sets
    model.T = Set(ordered=True, initialize=TimeData.index)  # Set for time steps
    model.B = Set(ordered=True, initialize=StorageData.index)  # Set for battery
    model.L = Set(ordered=True, initialize=LoadData.index)  # Set for loads
    model.P = Set(ordered=True, initialize=PVProduction.index)  # Set for PV
    model.E = Set(ordered=True, initialize=EVDemand.index)  # Set for EV load
    model.G = Set(ordered=True, initialize=TransformerData.index) # Set for (grid) transformer

#------------------------------------------------------------------
    #Define Parameters
    # Energy storage system
    model.BESS_Pmax = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_SOEmax = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_SOEini = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_Eff = Param(model.B, within=NonNegativeReals, mutable=True)
    # Load
    model.Consumption = Param(model.L, model.T, within=NonNegativeReals, mutable=True)  # Consumption of load j
    # PV parks
    model.PV = Param(model.P, model.T, within=NonNegativeReals, mutable=True)  # Production of PV system k
    # transformer
    model.Pmax = Param(model.G, within=NonNegativeReals, mutable=True)
    # moet hier nog -P bij voor energie terugvoeren?
    # timestep
    model.timestep = Param(model.T, within=NonNegativeReals, mutable=True)
    # EV... to do

#----------------------------------------------------------------
    # Initialize Parameters
    for b in model.B: #battery parameters
        model.BESS_Pmax[b] = StorageData.loc[b,'Pmax']
        model.BESS_SOEmax[b] = StorageData.loc[b,'SOEmax']
        model.BESS_SOEini[b] = StorageData.loc[b,'SOEini']
        model.BESS_Eff[b] = StorageData.loc[b,'Eff']

    for l in model.L: # load parameters
        for t in model.T:
            model.Consumption[l,t] = LoadData.loc[t,'LoadData']

    for p in model.P: # pv parameters
        for t in model.T:
            model.PV[p,t] = PVProduction.loc['PVProduction']

    for g in model.G: # transformer parameters
        model.Pmax[g] = TransformerData.loc[g,'Pmax']
        
    for t in model.T:
        model.timestep[t] = TimeData.loc[t, 'timestep']

    # to do: for e in model.E: (EV)

#--------------------------------------------------------
    # Define the Decision Variables
    #BESS
    model.SOE = Var(model.B, model.T, within=NonNegativeReals)
    model.Pch = Var(model.B, model.T, within=NonNegativeReals)
    model.Pdis = Var(model.B, model.T, within=NonNegativeReals)
    model.u_bess = Var(model.B, model.T, within=Binary)
    model.u_idle = Var(model.B, model.T, within=Binary)
    # to do: EV toevoegen

#---------------------------------------------------------
    # Define Constraints
    
    def ObjectiveFcn(model):
        return sum(sum(model.consumption[l,t] * model.timestep[t] for l in model.L) - sum(model.PV[p,t] * model.timestep[t] for p in model.P) - 
                   sum(model.Pdis[b,t]/model.BESS_Eff[b] for b in model.B) + sum(model.Pch[b,t]/model.BESS_Eff[b] for b in model.B) for t in model.T)
    
    def SOE(model, b, t):
        if model.T == 1:
            return model.SOE[b,t] == model.BESS_SOEini[b] + model.Pch[b,t] * model.BESS_Eff[b] - model.Pdis[b,t]/model.BESS_Eff[b]
        if model.T > 1:
            return model.SOE[b,t] == model.SOE[b, model.T.prev(t)] + model.Pch[b,t] * model.BESS_Eff[b] - model.Pdis[b,t]/model.BESS_Eff[b]
                  
    def BESS_SOE_max(model, b, t):
        return model.SOE[b,t] <= model.ESS_SOEmax[b]
    
    # BESS_SOE_min is not needed, since model.SOE specifies NonNegativeReals
    
    def BESS_Charging(model, b, t):
       return model.Pch[b,t] <= model.BESS_Pmax[b] * model.u_bess[b,t]

    def BESS_Discharging(model, b, t): 
       return model.Pdis[b, t] <= model.BESS_Pmax[b] * (1-model.u_bess[b, t])

    def BESS_idle(model, b, t):
        return model.u_bess[b,t] + model.u_idle[b,t] < 1
    
#----------------------------------------------------------
    # Add Constraints to the model
    
    model.Obj = Objective(rule=ObjectiveFcn)
    model.ConSOE = Constraint(model.B, model.T, rule=SOE)
    model.ConSOEmax = Constraint(model.B, model.T, rule=BESS_SOE_max)
    model.ConBESSCharging = Constraint(model.B, model.T, rule=BESS_Charging)
    model.ConBESSDischarging = Constraint(model.B, model.T, rule=BESS_Disharging)
    model.ConBESSidle = Constraint(model.B, model.T, rule=BESS_idle)
    
    return model

filename = 'variables_BAP.xlsx'
data = readInputFile(filename)

model = optimizationModel(data, 'test')
opt=SolverFactory('gurobi')
opt.options["MIPGap"] = 0.0
results=opt.solve(model)
