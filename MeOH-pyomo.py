# %%

"""
This is a pyomo dispatcher implementation of a basic Energy System (ES), composed of a PV (with possibility of curtailment),
a grid connection, an electrolyzer and a constant hydrogen off-take.
The dispatcher has been implemented as a function.
To be able to run the dispatcher I used the electricity price profile of zone SE3 in Sweden in year 2021 and a fictious fixed hydrogen price.
"""
from pyomo.core.base import Var, Constraint, Objective, NonNegativeReals, Reals, Binary, Integers, maximize
import pyomo.environ as pyo
from pyomo.core.kernel import value
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.base import ConcreteModel, RangeSet, Set, Param
import pyomo.environ as pyo

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% Here I define the dispatcher function


def dispatcher(design):

    # INITIALIZE the optimization framework
    m = ConcreteModel()

    # INITIALIZATION of SETS
    m.iIDX = Set(initialize=range(time_horizon))

    # INITIALIZATION of PARAMETERS

    """importing data in the pyomo framewrok: you have to convert lists and dataframes in dictionaries """

    # Power source
    m.P_PV = pyo.Param(
        m.iIDX, initialize=design["P_PV_data"]["P_PV_kW"].to_dict()
    )  # [kW] available power from the PV ## How to initialize this parameter?
    
    # Electrolyzer
    m.P_elec_min = pyo.Param(
        initialize=design["p_elec_min"]/100
    )  # [kW] minimum allowed power to electrolyzer
    m.P_elec_max = pyo.Param(
        initialize=design["p_elec_max"]/100
    )  # [kW] maximum allowed power to electrolyzer
    m.SEC_elec = pyo.Param(
        initialize=design["sec_elec_nominal"]) 

    # Reactor
    m.reactor_min = pyo.Param(
        initialize=design["Meoh_reactor_min"]/100
    )  # [kW] minimum allowed power to electrolyzer
    m.reactor_max = pyo.Param(
        initialize=design["Meoh_reactor_max"]/100
    )  # [kW] maximum allowed power to electrolyzer
    m.SEC_reactor = pyo.Param(
        initialize=design["sec_reactor_nominal"]
    )  # [kWh/kgH2] Specific Energy Consumption of the electrolyzer
  
    
    # Economic parameters
    m.scapex_reactor = pyo.Param(
         initialize=design["scapex_reactor"])
    m.scapex_elec = pyo.Param(
         initialize=design["scapex_elec"])
    
    m.el_price = pyo.Param(
        m.iIDX, initialize=design["Electricity_price_data"]["Grid_Price"].to_dict()
    )  # [€/kWh]
    m.meoh_price = pyo.Param(
        initialize=design["meoh_price_data"]
    )  # [€/kg] 


    # INITIALIZATION of VARIABLES

    # Operational planning 
    m.P_reactor = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kW] power supplied to reactor
    m.P_elec = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kW] power supplied to electrolyzer
    m.P_curt = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kW] power curtailed from the PV

    m.m_elec = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kg/h] hydrogen mass flow rate from electrolyzer
    m.m_meoh = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kg/h] methanol mass flow rate from reactor

    
    # Sizing variables
    m.reactor_size = pyo.Var(
     domain=pyo.NonNegativeReals
    )  # [kW] maximum flowrate of methanol from reactor
    m.chi_reactor = pyo.Var(
        m.iIDX, domain=pyo.Binary
    )  # [-] binary variable for the reactor operation

    m.elec_size = pyo.Var(
        domain=pyo.NonNegativeReals
    ) # [kW] maximum allowed power to electrolyzer
    m.chi_elec = pyo.Var(
        m.iIDX, domain=pyo.Binary
    )  # [-] binary variable for the electrolyzer operation

    
    # DEFINITION OF OPTIMIZATION PROBLEM ------------------------------------

    def obj_funct(m):
        capex = - m.scapex_elec * m.elec_size - m.scapex_reactor * m.reactor_size
        revenues = sum((m.meoh_price * m.m_meoh[i]) for i in m.iIDX)
        return capex + revenues

    m.obj = pyo.Objective(rule=obj_funct, sense=pyo.maximize)

    #DEFINITION OF CONSTRAINTS ----------------------------------------------

    # power equilibrium constraint *****
    def f_mass_equilibrium(m, i):
        return m.m_meoh[i] == m.m_elec [i] * (32/6)
    
    m.cstr_mass_equilibrium = pyo.Constraint( m.iIDX,rule=f_mass_equilibrium)
    
    def f_SEC_reactor(m, i):
        return m.m_meoh[i] == m.P_reactor[i] / m.SEC_reactor

    m.cstr_SEC_reactor = pyo.Constraint(m.iIDX, rule=f_SEC_reactor)

    def f_SEC_elec(m, i):
        return m.m_elec[i] == m.P_elec[i] / m.SEC_elec

    m.cstr_SEC_elec = pyo.Constraint(m.iIDX, rule=f_SEC_elec)

    def f_power_equilibrium(m, i):
        return m.P_elec[i] + m.P_curt[i] + m.P_reactor[i] == m.P_PV[i]

    m.cstr_power_equilibrium = pyo.Constraint(m.iIDX, rule=f_power_equilibrium)


    def f_reactor_max(m,i):
        return m.m_meoh[i] <= m.chi_reactor[i] * m.reactor_max * m.reactor_size

    m.cstr_reactor_max = pyo.Constraint(m.iIDX,rule=f_reactor_max)

    def f_reactor_min(m,i):
        return m.m_meoh[i] >= m.chi_reactor[i] * m.reactor_min * m.reactor_size

    m.cstr_reactor_min = pyo.Constraint( m.iIDX,rule=f_reactor_min)

    def f_electrolyzer_max(m, i):
        return m.P_elec[i] <= m.chi_elec[i] * m.P_elec_max * m.elec_size

    m.cstr_electrolyzer_max = pyo.Constraint(m.iIDX, rule=f_electrolyzer_max)

    def f_electrolyzer_min(m, i):
        return m.P_elec[i] >= m.chi_elec[i] * m.P_elec_min * m.elec_size

    m.cstr_electrolyzer_min = pyo.Constraint(m.iIDX, rule=f_electrolyzer_min)

    # selection of the optimization solver (minding that it is suited for the kind of problem)
    opt = pyo.SolverFactory("gurobi")
    #opt.set_instance(m)

    # resolution of the problem
    opt.solve(m)

    # Solve the model
    print("Solving model...")
    results = opt.solve(m, tee=False, keepfiles=False)                          
    # Change tee to True if you want to see solving status printed

    # Check is the problem is feasible
    status = results.solver.status
    termination_condition = results.solver.termination_condition
    print("\nSolver status:", status, termination_condition)

    data_time = np.array([i for i in m.iIDX])

    # Storing the data in arrays that can later be exported (maybe in a dataframe) and/or be displayed
    results = {}
    results["reactor_sz"] = np.array([pyo.value(m.reactor_size)])
    results["elec_sz"] = np.array([pyo.value(m.elec_size)])

    results["P_reactor"] = np.array([pyo.value(m.P_reactor[i]) for i in m.iIDX])
    results["P_elec"] = np.array([pyo.value(m.P_elec[i]) for i in m.iIDX])
    results["P_PV"] = np.array([pyo.value(m.P_PV[i]) for i in m.iIDX])
    results["P_curt"] = np.array([pyo.value(m.P_curt[i]) for i in m.iIDX])

    results["m_elec"] = np.array([pyo.value(m.m_elec[i]) for i in m.iIDX])
    results["m_meoh"] = np.array([pyo.value(m.m_meoh[i]) for i in m.iIDX])


    return (data_time, m, results) # return the other variables as well

from design import *


# %% I run the simulation
time_horizon = 8760 # how should we define this parameter? Years, Hours?
(data_time, m, results) = dispatcher(design)
# %%
# First subplot: Power output comparison
plt.subplot(2, 1, 1)  # (rows, columns, index)
plt.plot(data_time[:100], results["P_reactor"][:100], label='P_reactor', color='blue')
plt.plot(data_time[:100], results["P_elec"][:100], label='P_elec', color='green')
plt.plot(data_time[:100], results["P_curt"][:100], label='P_curt', color='red')
plt.plot(data_time[:100], results["P_PV"][:100], label='P_PV', color='black')

# Adding labels and title for the first subplot
plt.xlabel('Time')
plt.ylabel('Power')
plt.title('Power Output Comparison')
plt.legend()

# Second subplot: Methanol and Hydrogen mass
plt.subplot(2, 1, 2)  # Second plot
plt.plot(data_time[:100], results["m_meoh"][:100], label='Methanol Produced (m_meoh)', color='purple')
plt.plot(data_time[:100], results["m_elec"][:100], label='Hydrogen Consumed (m_elec)', color='orange')

# Adding labels and title for the second subplot
plt.xlabel('Time')
plt.ylabel('Mass')
plt.title('Methanol Produced and Hydrogen Consumed')
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

print(results["reactor_sz"])
print(results["elec_sz"])
# %%
