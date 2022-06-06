from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import pandas, numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

def readInputFile(filename):
    
    # Load all data from excel sheets
    
    LoadData = pandas.read_excel(filename, sheet_name= 'Load', index_col=0) #data of load entire neighbourhood
    TransformerData = pandas.read_excel(filename, sheet_name= 'Transformer', index_col=0) #rated power transformer
    PVProduction = pandas.read_excel(filename, sheet_name='PVProduction', index_col=0) # omgerekende irradiation
    StorageData = pandas.read_excel(filename, sheet_name= 'StorageSystem', index_col=0) # batterij
    EVDemand = pandas.read_excel(filename, sheet_name = 'EVDemand', index_col=0) #EV charging data

    # Return directory 
    return {'PVProduction':PVProduction, 'StorageData':StorageData,
            'LoadData':LoadData, 'TransformerData':TransformerData, 'EVDemand':EVDemand}

def optimizationModel(inputData, modelType):   
    
    # Unpack the data from the dictionary
    LoadData = inputData['LoadData']
    TransformerData = inputData['TransformerData']
    PVProduction = inputData['PVProduction']
    StorageData = inputData['StorageData']
    EVDemand = inputData['EVDemand']
#------------------------------------------------------------------------------
    # Define the Model
    model = ConcreteModel()

#------------------------------------------------------------------------------
    #Define Sets
    model.T = Set(ordered=True, initialize=LoadData.index)  # Set for time
    model.B = Set(ordered=True, initialize=StorageData.index)  # Set for battery
    model.G = Set(ordered=True, initialize=TransformerData.index) # Set for (grid) transformer

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
        # EV
    model.EV = Param(model.T, within=NonNegativeReals, mutable=True)

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

        # EV Parameter
    for t in model.T:
        model.EV[t] = EVDemand.loc[t, 'EVDemand']

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
    model.curtail_pv = Var(model.T, within=NonNegativeReals)    # Curtail % of PV
    model.PVprod = Var(model.T, within=NonNegativeReals)        # PV Production
    
        # Added for EV curtailment/different wattages
    model.curtail_ev = Var(model.T, within=NonNegativeReals)    # Curtail % of EV
    model.EVDelayed = Var(model.T, within=NonNegativeReals)
    model.EVTotal = Var(model.T, within=NonNegativeReals)
    model.EVSup = Var(model.T, within=NonNegativeReals)
    model.SOPSup = Var(model.T, within=NonNegativeReals)
    model.SOPDemand = Var(model.T, within=NonNegativeReals)

#------------------------------------------------------------------------------
    # Define Constraints
        # TODO: Add comments to this part explaining each constraint
   
    def CurtailEV(model, t):
        if model.T.ord(t) == len(model.T):
            return model.curtail_ev[t] == 1
        else:
            return model.curtail_ev[t] <= 1

    def TotalEVDemand(model, t):
        if model.T.ord(t) == 1:
            return model.EVTotal[t] == model.EV[t]
        if model.T.ord(t) > 1:
            return model.EVTotal[t] == model.EV[t] + model.EVDelayed[model.T.prev(t)]
        
    def SupplyEVNow(model, t):
        return model.EVSup[t] == model.EVTotal[t] * model.curtail_ev[t]
     
    def DelayEV(model, t):
        return model.EVDelayed[t] == model.EVTotal[t] - model.EVSup[t]
    
    def TotalSupply(model, t):       # Niet in mathematical model
        if model.T.ord(t) == 1:
            return model.SOPSup[t] == model.EVSup[t]
        if model.T.ord(t) > 1:
            return model.SOPSup[t] == model.SOPSup[model.T.prev(t)] + model.EVSup[t]
        
    def TotalDemand(model, t):      # Niet in mathematical model
        if model.T.ord(t) == 1:
            return model.SOPDemand[t] == model.EV[t]
        if model.T.ord(t) > 1:
            return model.SOPDemand[t] == model.SOPDemand[model.T.prev(t)] + model.EV[t]
    
    def ForceCharge(model, t):
        if model.T.ord(t) > 8:
            return model.SOPDemand[t-8] <= model.SOPSup[t]
        else:
            return Constraint.Skip      
        
    def ObjectiveFcn(model):
        return timestep*sum(model.Pgrid_plus[t] + model.Pgrid_minus[t] for t in model.T)    
        
    # Added in order to have curtailment
    def CurtailPV(model, t):
        return model.curtail_pv[t] <= 1 
    
    # Curtail > 0 not needed since its NonNegativeReal
    
    def PVcurtail(model, t):
        return model.PVprod[t] == model.PV[t] * model.curtail_pv[t]
    
    def PGrid(model, b, t):
        return model.Pgrid_plus[t] - model.Pgrid_minus[t] == model.Consumption[t] + model.Pch[b,t] \
                   + model.EVSup[t] - model.PVprod[t] - model.Pdis[b, t]
                   
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
    model.ConCurtailPV = Constraint(model.T, rule=CurtailPV)
    model.ConPVcurtail = Constraint(model.T, rule=PVcurtail)
    
    # Added in order to have curtailment of EV
    model.ConCurtailEV = Constraint(model.T, rule=CurtailEV)
    model.ConTotalEVDemand = Constraint(model.T, rule=TotalEVDemand)
    model.ConSupplyEVNow = Constraint(model.T, rule=SupplyEVNow)
    model.ConDelayEV = Constraint(model.T, rule=DelayEV)
    model.ConForceCharge = Constraint(model.T, rule=ForceCharge)
    model.ConTotalSupply = Constraint(model.T, rule=TotalSupply)
    model.ConTotalDemand = Constraint(model.T, rule=TotalDemand)
    
    return model

#------------------------------------------------------------------------------
    # Retrieve data from winter and summer Excel sheets and create model

data_winter = readInputFile('variables_BAP_winter.xlsx')
model_winter = optimizationModel(data_winter, 'winter') # Simulates the first week of January

data_summer = readInputFile('variables_BAP_summer.xlsx')
model_summer = optimizationModel(data_summer, 'summer') # Simulates the first week of July

#------------------------------------------------------------------------------
    # Initialises the solver and its settings and solves the model

opt=SolverFactory('gurobi')
opt.options["MIPGap"] = 0.0
opt.options['NonConvex'] = 2 # <= Solves the gurobi crash

results_winter=opt.solve(model_winter)
results_summer=opt.solve(model_summer)

#------------------------------------------------------------------------------
    # Extract data from both winter and summer models into arrays
       
# Retrieve simulation length from length of SOE array
x = []
for i in range(672):
    x.append(i * 0.25)

# Get Load plot data
    # Winter
load_plot_w = []
for t in model_winter.T:
    load_plot_w.append(model_winter.Consumption[t]() + model_winter.EV[t]())
    # Summer
load_plot_s = []
for t in model_summer.T:
    load_plot_s.append(model_summer.Consumption[t]() + model_summer.EV[t]())

# Get Solar plot data
    # Winter
solar_plot_w = []
for t in model_winter.T:
    solar_plot_w.append(model_winter.PV[t]())
    # Summer
solar_plot_s = []
for t in model_summer.T:
    solar_plot_s.append(model_summer.PV[t]())

#------------------------------------------------------------------------------
#     # Create the plots 
#         # For now this plot suffices to give a quick observation 
#             # Winter
# fig, axs_w = plt.subplots(2, 2)
# axs_w[0, 0].step(x, SOE_plot_w, color = 'green')
# axs_w[0, 0].set_title('SOE [kWh]')
# axs_w[0, 1].step(x, load_plot_w, color = 'blue')
# axs_w[0, 1].set_title('Load [kW]')
# axs_w[1, 0].step(x, solar_plot_w, color = 'orange')
# axs_w[1, 0].set_title('Possible Solar [kW]')
# axs_w[1, 1].step(x, solarprod_plot_w, color = 'red')
# axs_w[1, 1].set_title('Solar production [kW]')
# plt.suptitle('Winter')

#     # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs_w.flat:
#     ax.label_outer() 
    
#     # Summer
# fig, axs_s = plt.subplots(2, 2)
# axs_s[0, 0].step(x, SOE_plot_s, color = 'green')
# axs_s[0, 0].set_title('SOE [kWh]')
# axs_s[0, 1].step(x, load_plot_s, color = 'blue')
# axs_s[0, 1].set_title('Load [kW]')
# axs_s[1, 0].step(x, solar_plot_s, color = 'orange')
# axs_s[1, 0].set_title('Possible Solar [kW]')
# axs_s[1, 1].step(x, solarprod_plot_s, color = 'red')
# axs_s[1, 1].set_title('Solar production [kW]')
# plt.suptitle('Summer')

#     # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs_s.flat:
#     ax.label_outer()  
 
# #------------------------------------------------------------------------------
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
ax.set_xlabel("Time [h]")
# set y-axis label
ax.set_ylabel("Pgrid [kW]", color='blue')
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.step(x, Pgrid_plot_s, color="red")
ax2.set_ylabel("Pgrid [kW]",color="red")
plt.show()

#------------------------------------------------------------------------------
    # Printing the objective function to get total energy usage of the week
print("Winter with optimization:")
print(model_winter.Obj())
print('Winter no battery no PV:')
print(0.25*sum(load_plot_w))
print("Summer:")
print(model_summer.Obj())
print('Summer no battery no PV:')
print(0.25*sum(load_plot_s))

#-------------------
EV_total_w = []
for t in model_winter.T:
    EV_total_w.append(model_winter.EV[t]())
    
EV_sum_w = []
for t in model_winter.T:
    EV_sum_w.append(model_winter.EVSup[t]())
print('\n')
print(sum(EV_total_w))
print(sum(EV_sum_w))

EV_total_s = []
for t in model_summer.T:
    EV_total_s.append(model_summer.EV[t]())
    
EV_sum_s = []
for t in model_summer.T:
    EV_sum_s.append(model_summer.EVSup[t]())
print('\n')
print(sum(EV_total_s))
print(sum(EV_sum_s))

EV_demand_plot = []
for t in model_summer.T:
    EV_demand_plot.append(model_summer.EV[t]())
EV_supply_plot = []
for t in model_summer.T:
        EV_supply_plot.append(model_summer.EVSup[t]())

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
ax.step(x, EV_demand_plot, color="black", linewidth = 1, alpha=0.8)
ax.set_xlabel("Time [h]")
ax.set_ylabel("P [kW]", color='black')
ax.step(x, EV_supply_plot, color="blue", linewidth = 1, alpha=0.8)
plt.show()

#------------------------------------------------------------------------------
    # Create first plot, Load+SOE+PVprod

# Get total load plot data
    # Winter
Total_load_winter = []
for t in model_winter.T:
    Total_load_winter.append(model_winter.EVSup[t]()+model_winter.Consumption[t]())
    # Summer
Total_load_summer = []
for t in model_summer.T:
    Total_load_summer.append(model_summer.EVSup[t]()+model_summer.Consumption[t]())

# Get State of Energy plot data
    # Winter
SOE_plot_winter = []
for b in model_winter.B:
     for t in model_winter.T:
         SOE_plot_winter.append(model_winter.SOE[b, t]())
    # Summer
SOE_plot_summer = []
for b in model_summer.B:
     for t in model_summer.T:
         SOE_plot_summer.append(model_summer.SOE[b, t]()) 
         
# Get Curtailed Solar plot data
    # Winter
solarprod_plot_winter = []
for t in model_winter.T:
    solarprod_plot_winter.append(model_winter.PVprod[t]())
    # Summer
solarprod_plot_summer = []
for t in model_summer.T:
    solarprod_plot_summer.append(model_summer.PVprod[t]())
    
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
ax.step(x, Total_load_winter, color="blue", linewidth = 1.5, alpha=0.8, label = 'Total Load')
ax.step(x, solarprod_plot_winter, color="red", linewidth = 1.5, alpha=0.8, label = 'Used Solar Energy')
ax.set_xlabel("Time [h]")
ax.set_ylabel("P [kW]")
plt.minorticks_on()
plt.legend(bbox_to_anchor=(1.02, 0.3), loc='upper left', borderaxespad=0)
ax2=ax.twinx()
plt.minorticks_on()
ax2.step(x, SOE_plot_winter, color="green", linewidth = 1.5, label = 'BESS State of Energy')
ax2.set_ylabel("E [kWh]", color='green')
figure = plt.gcf()
figure.set_size_inches(35, 3)
plt.xticks(np.arange(0, 170, 5))
ax.margins(x=0.01)
plt.title("Load, Solar production and BESS SOE in 1st week of January", fontsize = 16)
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('winter_system.png', dpi=1000, bbox_inches='tight')
plt.show()      

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
ax.step(x, Total_load_summer, color="blue", linewidth = 1.5, alpha=0.8, label = 'Total Load')
ax.step(x, solarprod_plot_summer, color="red", linewidth = 1.5, alpha=0.8, label = 'Used Solar Energy')
ax.set_xlabel("Time [h]")
ax.set_ylabel("P [kW]")
plt.minorticks_on()
plt.legend(bbox_to_anchor=(1.02, 0.3), loc='upper left', borderaxespad=0)
ax2=ax.twinx()
plt.minorticks_on()
ax2.step(x, SOE_plot_summer, color="green", linewidth = 1.5, label = 'BESS State of Energy')
ax2.set_ylabel("E [kWh]", color='green')
figure = plt.gcf()
figure.set_size_inches(35, 3)
plt.xticks(np.arange(0, 170, 5))
ax.margins(x=0.01)
plt.title("Load, Solar production and BESS SOE in 1st week of July", fontsize = 16)
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('summer_system.png', dpi=1000, bbox_inches='tight')
plt.show()  

# # Show the major grid lines with dark grey lines
# plt.grid(axis = 'both', which='major', color='#666666', linestyle='-', alpha=0)

# # Show the minor grid lines with very faint and almost transparent grey lines
# plt.minorticks_on()
# plt.grid(axis = 'both',which='minor', color='#999999', linestyle='-', alpha=0)
#------------------------------------------------------------------------------
    # Get Total Curtail Percentage Per Day In Summer
get_curtail_summer = []
for t in model_summer.T:
    get_curtail_summer.append((1-model_summer.curtail_pv[t]())*15)     # Gives all curtail values in minutes/15 minutes
  
for i in range(len(solar_plot_s)):
    if solar_plot_s[i] == 0: 
        get_curtail_summer[i] = 0
    
get_curtail_summer_per_day = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
    for i in range(96):
        get_curtail_summer_per_day[j] += get_curtail_summer[j*96+i]
        
minutes_day_solar_summer = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
    for i in range(96):
        if solar_plot_s[j*96+i] != 0:
            minutes_day_solar_summer[j] += 15

get_curtail_percentage_summer = [0, 0, 0, 0, 0, 0, 0]
for i in range(7):
    get_curtail_percentage_summer[i] = get_curtail_summer_per_day[i]/minutes_day_solar_summer[i] # Divide by minutes that day of solar energy
#------------------------------------------------------------------------------
    # Get Total Curtail Percentage Per Day In Winter
get_curtail_winter = []
for t in model_winter.T:
    get_curtail_winter.append((1-model_winter.curtail_pv[t]())*15)     # Gives all not used minutes/15 minutes
  
for i in range(len(solar_plot_s)):
    if solar_plot_w[i] == 0: 
        get_curtail_winter[i] = 0
    
get_curtail_winter_per_day = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
    for i in range(96):
        get_curtail_winter_per_day[j] += get_curtail_winter[j*96+i]
        
minutes_day_solar_winter = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
    for i in range(96):
        if solar_plot_w[j*96+i] != 0:
            minutes_day_solar_winter[j] += 15

get_curtail_percentage_winter = [0, 0, 0, 0, 0, 0, 0]
for i in range(7):
    get_curtail_percentage_winter[i] = get_curtail_winter_per_day[i]/minutes_day_solar_winter[i] # Divide by minutes that day of solar energy
#------------------------------------------------------------------------------
    # Plot the bar chart of solar curtailment
Solar_energy_summer = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
     for i in range(96):
         Solar_energy_summer[j] += solar_plot_s[j*96+i]*0.25
    
Used_solar_energy_summer = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
     for i in range(96):
         Used_solar_energy_summer[j] += solarprod_plot_summer[j*96+i]*0.25
         
Solar_energy_winter = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
     for i in range(96):
         Solar_energy_winter[j] += solar_plot_w[j*96+i]*0.25         

Used_solar_energy_winter = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
     for i in range(96):
         Used_solar_energy_winter[j] += solarprod_plot_winter[j*96+i]*0.25         
         
Used_solar_energy_summer_curtail = [0, 0, 0, 0, 0, 0, 0]
for j in range(7):
    Used_solar_energy_summer_curtail[j] = Solar_energy_summer[j]*get_curtail_percentage_summer[j]
    
X = ['July 1st','July 2nd','July 3rd','July 4th', 'July 5th', 'July 6th', 'July 7th']
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Solar_energy_summer, 0.4, label = 'Total Solar Energy', color = 'orange')
plt.bar(X_axis + 0.2, Used_solar_energy_summer, 0.4, label = 'Used Solar Energy', color = 'blue')
  
plt.xticks(X_axis, X)
plt.xlabel("Days")
plt.ylabel("Energy [kWh]")
plt.title("Solar curtailment during summer")
plt.legend()
plt.show()

X1 = ['Jan 1st','Jan 2nd','Jan 3rd','Jan 4th', 'Jan 5th', 'Jan 6th', 'Jan 7th']
X1_axis = np.arange(len(X1))

plt.bar(X1_axis - 0.2, Solar_energy_winter, 0.4, label = 'Total Solar Energy', color = 'orange')
plt.bar(X1_axis + 0.2, Used_solar_energy_winter, 0.4, label = 'Used Solar Energy', color = 'blue')
  
plt.xticks(X1_axis, X1)
plt.xlabel("Days")
plt.ylabel("Energy [kWh]")
plt.title("Solar curtailment during winter")
plt.legend()
plt.show()

plt.bar(X_axis, get_curtail_percentage_summer, 0.75, label = 'Curtail Energy', color = 'orange')
  
plt.xticks(X_axis, X)
plt.xlabel("Days")
plt.ylabel("Percentage of curtailment per day")
plt.title("Solar Energy Curtailment")
plt.show()

