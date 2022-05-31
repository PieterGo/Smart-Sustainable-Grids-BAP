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
#------------------------------------------------------------------------------
    # Define the Model
    model = ConcreteModel()

#------------------------------------------------------------------------------
    #Define Sets
    model.T = Set(ordered=True, initialize=LoadData.index)  # Set for time
    model.B = Set(ordered=True, initialize=StorageData.index)  # Set for battery
    model.P = Set(ordered=True, initialize=PVProduction.index)  # Set for PV
    model.E = Set(ordered=True, initialize=EVDemand.index)  # Set for EV load
    model.G = Set(ordered=True, initialize=TransformerData.index) # Set for (grid) transformer
    model.X = Set(ordered=True, initialize=TimeStep.index) # Time Step

#------------------------------------------------------------------------------
    #Define Parameters
        # Battery Energy Storage System
    model.BESS_Pmax = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_SOEmax = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_SOEini = Param(model.B, within=NonNegativeReals, mutable=True)
    model.BESS_Eff = Param(model.B, within=NonNegativeReals, mutable=True)
        # Load
    model.Consumption = Param(model.T, within=NonNegativeReals, mutable=True)  # Consumption of load j
        # PV Generation
    model.PV = Param(model.T, within=NonNegativeReals, mutable=True)  # Production of PV system k
        # Transformer
    model.Pmax = Param(model.G, within=NonNegativeReals, mutable=True)
    # EV... to do

#------------------------------------------------------------------------------
    # Initialize Parameters
        # BESS Parameters
    for b in model.B:
        model.BESS_Pmax[b] = StorageData.loc[b,'Pmax']
        model.BESS_SOEmax[b] = StorageData.loc[b,'SOEmax']
        model.BESS_SOEini[b] = StorageData.loc[b,'SOEini']
        model.BESS_Eff[b] = StorageData.loc[b,'Eff']

        # PV generation Parameter
    for t in model.T:
        model.PV[t] = PVProduction.loc[t,'PVProduction']

        # Transformer Parameter
    for g in model.G:
        model.Pmax[g] = TransformerData.loc[g,'Pmax']
        
       # The timestep of the model 
    timestep = 0.25
        
        # Load Parameter
    for t in model.T:
        model.Consumption[t] = LoadData.loc[t,'LoadData']

    # to do: for e in model.E: (EV)

#------------------------------------------------------------------------------
    # Define the Decision Variables
        # BESS
    model.SOE = Var(model.B, model.T, within=NonNegativeReals)  # SOE
    model.Pch = Var(model.B, model.T, within=NonNegativeReals)  # P charge
    model.Pdis = Var(model.B, model.T, within=NonNegativeReals) # P discharge
    model.u_bess = Var(model.B, model.T, within=Binary)         # 1 is charging
    model.u_idle = Var(model.B, model.T, within=Binary)         # 1 is idleing
        # Transformer
    model.Pgrid_plus = Var(model.T, within=NonNegativeReals)    # pushing from
    model.Pgrid_minus = Var(model.T, within=NonNegativeReals)   # pulling to
    model.u_grid = Var(model.T, within=Binary)                  # 1 is pulling
    
    # Added in order to have curtailment
    model.u_curtail = Var(model.T, within=NonNegativeReals)     # Curtail %
    model.PVprod = Var(model.T, within=NonNegativeReals)        # PV Production
    
    # to do: EV toevoegen

#------------------------------------------------------------------------------
    # Define Constraints
        # TODO: Add comments to this part explaining each constraint
        
    # Added in order to have curtailment
    def Curtail(model, t):
        return model.u_curtail[t] <= 1 
    
    # Curtail > 0 not needed since its NonNegativeReal
    
    def PVcurtail(model, t):
        return model.PVprod[t] == model.PV[t] * model.u_curtail[t]
      
    def ObjectiveFcn(model):
        return timestep*sum(model.Pgrid_plus[t] + model.Pgrid_minus[t] for t in model.T)
    
    def PGrid(model, b, t):
        return model.Pgrid_plus[t] - model.Pgrid_minus[t] == model.Consumption[t] + model.Pch[b,t] \
                   - model.PVprod[t] - model.Pdis[b, t]
                   
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
    
    
#------------------------------------------------------------------------------
    # Add Constraints to the model
        # TODO: Add comments to this
    
    model.Obj = Objective(rule=ObjectiveFcn)
    model.ConPGrid = Constraint(model.B, model.T, rule=PGrid)
    model.ConGridPull = Constraint(model.G, model.T, rule=GridPull)
    model.ConGridPush = Constraint(model.G, model.T, rule=GridPush)
    model.ConSOE = Constraint(model.B, model.T, rule=SOE)
    model.ConSOEmax = Constraint(model.B, model.T, rule=BESS_SOE_max)
    model.ConBESSCharging = Constraint(model.B, model.T, rule=BESS_Charging)
    model.ConBESSDischarging = Constraint(model.B, model.T, rule=BESS_Discharging)
    model.ConBESSidle = Constraint(model.B, model.T, rule=BESS_idle)
    
    # Added in order to have curtailment
    model.ConCurtail = Constraint(model.T, rule=Curtail)
    model.ConPVcurtail = Constraint(model.T, rule=PVcurtail)
    
    return model

#------------------------------------------------------------------------------
    # Retrieve data from winter and summer Excel sheets and create model

data_winter = readInputFile('variables_BAP_winter.xlsx')
model_winter = optimizationModel(data, 'winter') # Simulates the first week of January

data_summer = readInputFile('variables_BAP_summer.xlsx')
model_summer = optimizationModel(data, 'summer') # Simulates the first week of July

#------------------------------------------------------------------------------
    # Initialises the solver and its settings and solves the model

opt=SolverFactory('gurobi')
opt.options["MIPGap"] = 0.0

results=opt.solve(model1)

#------------------------------------------------------------------------------
    # Extract data from both winter and summer models into arrays

# Get State of Energy plot data
    # Winter
SOE_plot_w = []
for b in model_winter.B:
     for t in model_winter.T:
         SOE_plot_w.append(model_winter.SOE[b, t]())
    # Summer
SOE_plot_s = []
for b in model_summer.B:
     for t in model_summer.T:
         SOE_plot_s.append(model_summer.SOE[b, t]()) 
       
# Retrieve simulation length from length of SOE array
x = range(len(SOE_plot_summer)) 

# Get Load plot data
    # Winter
load_plot_w = []
for t in model_winter.T:
    load_plot_w.append(model_winter.Consumption[t]())
    # Summer
load_plot_s = []
for t in model_summer.T:
    load_plot_s.append(model_summer.Consumption[t]())

# Get Solar plot data
    # Winter
solar_plot_w = []
for t in model_winter.T:
    solar_plot_w.append(model_winter.PV[t]())
    # Summer
solar_plot_s = []
for t in model_summer.T:
    solar_plot_s.append(model_summer.PV[t]())

# Get Curtailed Solar plot data
    # Winter
solarprod_w_plot = []
for t in model_winter.T:
    solarprod_w_plot.append(model_winter.PVprod[t]())
    # Summer
solarprod_s_plot = []
for t in model_summer.T:
    solarprod_s_plot.append(model_summer.PVprod[t]())

#------------------------------------------------------------------------------
    # Create the plots 
        # For now this plot suffices to give a quick observation 
            # Winter
fig, axs_w = plt.subplots(2, 2)
axs_w[0, 0].step(x, SOE_plot_w, color = 'blue')
axs_w[0, 0].set_title('SOE [kWh]')
axs_w[0, 1].step(x, load_plot_w, color = 'blue')
axs_w[0, 1].set_title('Load [kW]')
axs_w[1, 0].step(x, solar_plot_w, color = 'blue')
axs_w[1, 0].set_title('Possible Solar [kW]')
axs_w[1, 1].step(x, solarprod_plot_w, color = 'blue')
axs_w[1, 1].set_title('Solar production [kW]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs_w.flat:
    ax.label_outer() 
    
    # Summer
fig, axs_s = plt.subplots(2, 2)
axs_s[0, 0].step(x, SOE_plot_s, color = 'red')
axs_s[0, 0].set_title('SOE [kWh]')
axs_s[0, 1].step(x, load_plot_s, color = 'red')
axs_s[0, 1].set_title('Load [kW]')
axs_s[1, 0].step(x, solar_plot_s, color = 'red')
axs_s[1, 0].set_title('Possible Solar [kW]')
axs_s[1, 1].step(x, solarprod_plot_s, color = 'red')
axs_s[1, 1].set_title('Solar production [kW]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs_s.flat:
    ax.label_outer()  
 
#------------------------------------------------------------------------------
    # Get data from transformer and plot for both winter and summer
  
# Get Energy pushed to grid
    # Winter
Pgrid_plot_w = []
for t in model_winter.T:
    Pgrid_plot_w.append(-0.25*model_winter.Pgrid_minus[t]()+0.25*model_winter.Pgrid_plus[t]())
    # Summer
Pgrid_plot_s = []
for t in model_summer.T:
    Pgrid_plot_s.append(-0.25*model_summer.Pgrid_minus[t]()+0.25*model_summer.Pgrid_plus[t]())
     
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.step(x, Pgrid_plot_w, color="blue")
# set x-axis label
ax.set_xlabel("timestep")
# set y-axis label
ax.set_ylabel("Pgrid [kW]")
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.step(x, Pgrid_plot_s, color="red")
ax2.set_ylabel("Pgrid [kW]",color="red")
plt.show()
    
#------------------------------------------------------------------------------
    # Printing the objective function to get total energy usage of the week
print(model1.Obj())
