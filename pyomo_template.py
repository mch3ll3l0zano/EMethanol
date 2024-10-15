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
    m = pyo.ConcreteModel()

    # INITIALIZATION of SETS
    # m.iIDX is the time index, which set the optimization horizon of the problem in an hourly fashion
    m.iIDX = pyo.Set(initialize=range(time_horizon))

    # INITIALIZATION of PARAMETERS

    """importing data in the pyomo framewrok: you have to convert lists and dataframes in dictionaries """
    m.el_price = pyo.Param(
        m.iIDX, initialize=design["Electricity_price_data"]["Grid_Price"].to_dict()
    )  # [€/kWh] ## How to initialize this parameter?
    m.h2_price = pyo.Param(
        initialize=design["Hydrogen_price_data"]
    )  # [€/kg] #this does not have an index its ust a floting point number so its fine
    m.P_PV = pyo.Param(
        m.iIDX, initialize=design["P_PV_data"]["P_PV_kW"].to_dict()
    )  # [kW] available power from the PV ## How to initialize this parameter?
    m.P_elec_min = pyo.Param(
        initialize=design["p_elec_min"]/100
    )  # [kW] minimum allowed power to electrolyzer
    m.P_elec_max = pyo.Param(
        initialize=design["p_elec_max"]/100
    )  # [kW] maximum allowed power to electrolyzer
    m.SEC_elec = pyo.Param(
        initialize=design["sec_elec_nominal"]
    )  # [kWh/kgH2] Specific Energy Consumption of the electrolyzer
    m.m_H2_demand = pyo.Param(
        initialize=design["Hydrogen_demand"]
    )  # [kg/h] constant hydrogen demand per hour

    m.V_sto_max = pyo.Param(
        initialize=design["V_sto_max"]/100
    )  # [kg] maximum storage capacity
    m.V_sto_min = pyo.Param(
        initialize=design["V_sto_min"]/100
    )  # [kg] minimum storage capacity

    # then initialize the other parameters in the same way ## Partially DONE

    # INITIALIZATION of VARIABLES

    m.P_PV_to_elec = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kW] power from the PV to the electrolyzer
    m.P_elec = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kW] power supplied to electrolyzer
    m.P_curt = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kW] power curtailed from the PV

    m.chi_elec = pyo.Var(
        m.iIDX, domain=pyo.Binary
    )  # [-] binary variable for the electrolyzer operation

    m.m_elec = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kg/h] hydrogen mass flow rate from electrolyzer
    m.elec_size = pyo.Var(
        domain=pyo.NonNegativeReals)  # [kW] size of the electrolyzer
    
    m.stored_H2 = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kg] stored hydrogen in the storage tank
    m.m_sto_in = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kg/h] hydrogen mass flow rate into the storage tank
    m.m_sto_out = pyo.Var(
        m.iIDX, domain=pyo.NonNegativeReals
    )  # [kg/h] hydrogen mass flow rate out of the storage tank
    m.chi_sto = pyo.Var(
        m.iIDX, domain=pyo.Binary
    )  # [-] binary variable for the storage tank operation
    m.sto_size = pyo.Var(
        domain=pyo.NonNegativeReals
    )  # [kg] size of the storage tank

    # initialize the other variables in the same way and mind if it binary negative or positive real etc ## DONE

    # DEFINITION of the optimization problem

    # definition of the objective function
    def obj_funct(m):
        capex = -m.elec_size * 2000 - m.sto_size * 3000  # [€/kW] cost of the electrolyzer
        opex = sum(
            (m.h2_price * m.m_elec[i]) for i in m.iIDX
        )
        return capex + opex

    m.obj = pyo.Objective(rule=obj_funct, sense=pyo.maximize)

    # power equilibrium constraint
    def f_power_equilibrium(m, i):
        return m.P_PV_to_elec[i] == m.P_elec[i]

    m.cstr_power_equilibrium = pyo.Constraint(m.iIDX, rule=f_power_equilibrium)

    def f_power_curtailment(m, i):
        return m.P_PV_to_elec[i] + m.P_curt[i] == m.P_PV[i]

    m.cstr_power_curtailment = pyo.Constraint(m.iIDX, rule=f_power_curtailment)

    def f_electrolyzer_max(m, i):
        return m.P_elec[i] <= m.chi_elec[i] * m.P_elec_max * m.elec_size

    m.cstr_electrolyzer_max = pyo.Constraint(m.iIDX, rule=f_electrolyzer_max)

    def f_electrolyzer_min(m, i):
        return m.P_elec[i] >= m.chi_elec[i] * m.P_elec_min * m.elec_size 

    m.cstr_electrolyzer_min = pyo.Constraint(m.iIDX, rule=f_electrolyzer_min)

    def storage_level_rule(m, i):
        if i>1:
            # Initial storage level
            return (m.stored_H2[i] == m.stored_H2[i-1]+
                        ((m.m_sto_in[i-1])-
                         (m.m_sto_out[i-1]))/14.5
                    )
        else:
            # Energy balance constraint
            return m.stored_H2[i] == 0.5 * m.sto_size
        
    m.cstr_storage_level = pyo.Constraint(m.iIDX, rule=storage_level_rule)

    def f_storage_max(m, i):
        return m.stored_H2[i] <= m.chi_sto[i] * m.V_sto_max * m.sto_size

    m.cstr_storage_max = pyo.Constraint(m.iIDX, rule=f_storage_max)

    def f_storage_min(m, i):
        return m.stored_H2[i] >= m.chi_sto[i] * m.V_sto_min * m.sto_size 

    m.cstr_storage_min = pyo.Constraint(m.iIDX, rule=f_storage_min)

    def f_SEC(m, i):
        return m.m_elec[i] + m.m_sto_in[i] == m.P_elec[i] / m.SEC_elec

    m.cstr_SEC = pyo.Constraint(m.iIDX, rule=f_SEC)

    def f_constant_H2_demand(m, i):
        return m.m_elec[i] + m.m_sto_out[i] == m.m_H2_demand

    m.cstr_constant_H2_demand = pyo.Constraint(m.iIDX, rule=f_constant_H2_demand)

    # write the other constraints

    # selection of the optimization solver (minding that it is suited for the kind of problem)
    opt = pyo.SolverFactory("gurobi")

    # resolution of the problem
    opt.solve(m)

    data_time = np.array([i for i in m.iIDX])

    # Storing the data in arrays that can later be exported (maybe in a dataframe) and/or be displayed
    elec_size = np.array([pyo.value(m.elec_size)])
    sto_size = np.array([pyo.value(m.sto_size)])
    P_elec = np.array([pyo.value(m.P_elec[i]) for i in m.iIDX])
    #P_grid = np.array([pyo.value(m.P_grid[i]) for i in m.iIDX])
    P_PV_to_elec = np.array([pyo.value(m.P_PV_to_elec[i]) for i in m.iIDX])
    #P_curt = np.array([pyo.value(m.P_curt[i]) for i in m.iIDX])
    chi_elec = np.array([pyo.value(m.chi_elec[i]) for i in m.iIDX])
    m_elec = np.array([pyo.value(m.m_elec[i]) for i in m.iIDX])
    m_sto_in = np.array([pyo.value(m.m_sto_in[i]) for i in m.iIDX])
    m_sto_out = np.array([pyo.value(m.m_sto_out[i]) for i in m.iIDX])
    
    stored_H2 = np.array([pyo.value(m.stored_H2[i]) for i in m.iIDX])
    chi_elec = np.array([pyo.value(m.chi_elec[i]) for i in m.iIDX])
    # read the other variables in the same way

    return (data_time, m, m_elec, m_sto_in, m_sto_out, 
            P_elec, P_PV_to_elec, 
            elec_size, sto_size, stored_H2)  # return the other variables as well

from design import *

# %% I run the simulation
time_horizon = 8760 # how should we define this parameter? Years, Hours?
(data_time, m, m_elec, m_sto_in, m_sto_out, P_elec, 
 P_PV_to_elec, elec_size, sto_size, stored_H2) = dispatcher(design)

# %%
# First subplot: Power output comparison
plt.subplot(2, 1, 1)  # (rows, columns, index)
plt.plot(data_time[:100], m_elec[:100], label='m_elec', color='blue')
plt.plot(data_time[:100], m_sto_in[:100], label='m_sto_in', color='green')
plt.plot(data_time[:100], m_sto_out[:100], label='m_sto_out', color='red')

# Adding labels and title for the first subplot
plt.xlabel('Time')
plt.ylabel('Mass Flow Rate')
plt.title('Mass Flow Rate Comparison')
plt.legend()

# Second subplot: Methanol and Hydrogen mass
plt.subplot(2, 1, 2)  # Second plot
plt.plot(data_time[:100], P_elec[:100], label='Electrolyzers Power', color='purple')
plt.plot(data_time[:100], P_PV_data[:100]["P_PV_kW"], label='PV', color='orange')

# Adding labels and title for the second subplot
plt.xlabel('Time')
plt.ylabel('Power [kW]')
plt.title('Power Output Comparison')
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

# %%
