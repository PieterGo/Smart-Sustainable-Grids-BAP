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
    
    TimeStep = pandas.read_excel(filename, sheet_name='Timestep', index_col=0) #time step
    TimeData = pandas.read_excel(filename, sheet_name='Time', index_col=0) #time data 
    LoadData = pandas.read_excel(filename, sheet_name= 'Load', index_col=0) #data of load entire neighbourhood
    TransformerData = pandas.read_excel(filename, sheet_name= 'Transformer', index_col=0) #rated power transformer
    PVProduction = pandas.read_excel(filename, sheet_name='PVProduction', index_col=0) # omgerekende irradiation
    StorageData = pandas.read_excel(filename, sheet_name= 'StorageSystem', index_col=0) # batterij
    EVDemand= pandas.read_excel(filename, sheet_name = 'EVDemand', index_col=0) #EV charging data

    # Return directory 
    return {'PVProduction':PVProduction, 'StorageData':StorageData, 'TimeData':TimeData, 'TimeStep':TimeStep,
            'LoadData':LoadData, 'TransformerData':TransformerData, 'EVDemand':EVDemand}

def optimizationModel(inputData, modelType):   
    
    # Unpack the data from the dictionary
    LoadData = inputData['LoadData']
    TransformerData = inputData['TransformerData']
    PVProduction = inputData['PVProduction']
    StorageData = inputData['StorageData']
    EVDemand = inputData['EVDemand']
    TimeData = inputData['TimeData']
    TimeStep = inputData['TimeStep']
#-------------------------------------------------------------------
    # Define the Model
    model = ConcreteModel()

#------------------------------------------------------------------
    #Define Sets
    #model.T = Set(ordered=True, initialize=TimeData.index)  # Set for time
    model.T = Set(ordered=True, initialize=LoadData.index)  # Set for time
    model.B = Set(ordered=True, initialize=StorageData.index)  # Set for battery
    #model.L = Set(ordered=True, initialize=LoadData.index)  # Set for loads
    model.P = Set(ordered=True, initialize=PVProduction.index)  # Set for PV
    model.E = Set(ordered=True, initialize=EVDemand.index)  # Set for EV load
    model.G = Set(ordered=True, initialize=TransformerData.index) # Set for (grid) transformer
    model.X = Set(ordered=True, initialize=TimeStep.index) # Time Step

#------------------------------------------------------------------
    #Define Parameters
    # Energy storage system
    model.BESS_Pmax = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_SOEmax = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_SOEini = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_Eff = Param(model.B, within=NonNegativeReals, mutable=True)
    # Load
    model.Consumption = Param(model.T, within=NonNegativeReals, mutable=True)  # Consumption of load j
    # PV parks
    model.PV = Param(model.T, within=NonNegativeReals, mutable=True)  # Production of PV system k
    # transformer
    model.Pmax = Param(model.G, within=NonNegativeReals, mutable=True)
    # moet hier nog -P bij voor energie terugvoeren?
    # timestep
    model.timestep = Param(model.X, within=NonNegativeReals, mutable=True)
    # EV... to do

#----------------------------------------------------------------
    # Initialize Parameters
    for b in model.B: #battery parameters
        model.BESS_Pmax[b] = StorageData.loc[b,'Pmax']
        model.BESS_SOEmax[b] = StorageData.loc[b,'SOEmax']
        model.BESS_SOEini[b] = StorageData.loc[b,'SOEini']
        model.BESS_Eff[b] = StorageData.loc[b,'Eff']

    for t in model.T: # pv parameters
        model.PV[t] = PVProduction.loc[t,'PVProduction']

    for g in model.G: # transformer parameters
        model.Pmax[g] = TransformerData.loc[g,'Pmax']
        
    #for x in model.X:
        #model.timestep[x] = TimeStep.loc[x, 'timestep']
        timestep = 0.25
        
    #for l in model.L: # load parameters
    for t in model.T:
        model.Consumption[t] = LoadData.loc[t,'LoadData']

    # to do: for e in model.E: (EV)

#------------------------------------------------------------------------------
    # Define the Decision Variables
    #BESS
    model.SOE = Var(model.B, model.T, within=NonNegativeReals)
    model.Pch = Var(model.B, model.T, within=NonNegativeReals)
    model.Pdis = Var(model.B, model.T, within=NonNegativeReals)
    model.u_bess = Var(model.B, model.T, within=Binary)         # 1 is charging
    model.u_idle = Var(model.B, model.T, within=Binary)
    model.Pgrid_plus = Var(model.T, within=NonNegativeReals)    # pushing from
    model.Pgrid_minus = Var(model.T, within=NonNegativeReals)   # pulling to
    model.u_grid = Var(model.T, within=Binary)                  # 1 is pulling
    # to do: EV toevoegen

#---------------------------------------------------------
    # Define Constraints
      
    def ObjectiveFcn(model):
        return timestep*sum(model.Pgrid_plus[t] + model.Pgrid_minus[t] for t in model.T)
    
    def PGrid(model, b, t):
        return model.Pgrid_plus[t] - model.Pgrid_minus[t] == model.Consumption[t] + model.Pch[b,t] \
                   - model.PV[t] - model.Pdis[b, t]
                   
    def GridPull(model, g, t):
        return model.Pgrid_minus[t] <= model.Pmax[g] * model.u_grid[t]
  
    def GridPush(model, g, t):
        return model.Pgrid_plus[t] <= model.Pmax[g] * (1-model.u_grid[t])


    def SOE(model, b, t):
        if model.T.ord(t) == 1:
            return model.SOE[b,t] == model.BESS_SOEini[b] + timestep * (model.Pch[b,t] \
                                    * model.BESS_Eff[b] - model.Pdis[b,t]/model.BESS_Eff[b])
        if model.T.ord(t) > 1:
            return model.SOE[b,t] == model.SOE[b, model.T.prev(t)] + timestep * (model.Pch[b,t] \
                                    * model.BESS_Eff[b] - model.Pdis[b,t]/model.BESS_Eff[b])

    def BESS_SOE_max(model, b, t):
        return model.SOE[b,t] <= model.BESS_SOEmax[b]
    
    # BESS_SOE_min is not needed, since model.SOE specifies NonNegativeReals
    
    def BESS_Charging(model, b, t):
        return model.Pch[b,t] <=  model.BESS_Pmax[b] * model.u_bess[b, t]
                                                        
    def BESS_Discharging(model, b, t): 
       return model.Pdis[b, t] <= model.BESS_Pmax[b] * (1-model.u_bess[b, t])

    def BESS_idle(model, b, t):
        return model.u_bess[b,t] + model.u_idle[b,t] <= 1
    
    
#----------------------------------------------------------
    # Add Constraints to the model
    
    model.Obj = Objective(rule=ObjectiveFcn)
    model.ConPGrid = Constraint(model.B, model.T, rule=PGrid)
    model.ConGridPull = Constraint(model.G, model.T, rule=GridPull)
    model.ConGridPush = Constraint(model.G, model.T, rule=GridPush)
    model.ConSOE = Constraint(model.B, model.T, rule=SOE)
    model.ConSOEmax = Constraint(model.B, model.T, rule=BESS_SOE_max)
    model.ConBESSCharging = Constraint(model.B, model.T, rule=BESS_Charging)
    model.ConBESSDischarging = Constraint(model.B, model.T, rule=BESS_Discharging)
    model.ConBESSidle = Constraint(model.B, model.T, rule=BESS_idle)
    
    return model

filename = 'variables_BAP.xlsx'
data = readInputFile(filename)

model1 = optimizationModel(data, 'test')
# solver settings
opt=SolverFactory('gurobi')
opt.options["MIPGap"] = 0.0

results=opt.solve(model1)
#model1.display()
#model1.pprint()


# Get SOE plot data
SOE_plot = []
for b in model1.B:
     for t in model1.T:
         SOE_plot.append(model1.SOE[b, t]())
       
# Create X-axis
x = range(len(SOE_plot)) 

# Get Load plot data
load_plot = []
for t in model1.T:
    load_plot.append(model1.Consumption[t]())

# Get PV plot data
solar_plot = []
for t in model1.T:
    solar_plot.append(model1.PV[t]())

# Get Transformer plot data
Obj_plot = []
for t in model1.T:
    Obj_plot.append(model1.Pgrid_plus[t]() + model1.Pgrid_minus[t]())

fig, axs = plt.subplots(2, 2)
axs[0, 0].step(x, SOE_plot, color = 'red')
axs[0, 0].set_title('SOE [kWh]')
axs[0, 1].step(x, load_plot, color = 'blue')
axs[0, 1].set_title('Load [kW]')
axs[1, 0].step(x, solar_plot, color = 'orange')
axs[1, 0].set_title('Solar [kW]')
axs[1, 1].step(x, Obj_plot, color = 'black')
axs[1, 1].set_title('Transformer [kW]')

for ax in axs.flat:
    ax.set(xlabel='timestep', ylabel='y-label')

#Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()    
    
# Get Energy pushed to grid
Pgrid_plot = []
for t in model1.T:
    Pgrid_plot.append(-0.25*model1.Pgrid_minus[t]()+0.25*model1.Pgrid_plus[t]())
    
pull_from = []
for t in model1.T:
    pull_from.append(0.25*model1.Pgrid_plus[t]())
    
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.set_ylim(min(Pgrid_plot)-1, -1*min(Pgrid_plot))
ax.step(x, Pgrid_plot, color="blue")
# set x-axis label
ax.set_xlabel("timestep")
# set y-axis label
ax.set_ylabel("Pgrid [kW]")
# twin object for two different y-axis on the sample plot
#ax2=ax.twinx()
# make a plot with different y-axis using second axis object
#ax2.set_ylim(-10, 10)
#ax2.step(x, pull_from, color="blue")
#ax2.set_ylabel("Pgrid [kW]",color="blue")
plt.show()
    



# Printing the objective function
print(model1.Obj())
