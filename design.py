import datetime
import pandas as pd

P_PV_data = pd.read_csv(
    "PVdata_1MWp_1year_unit_kW.csv", names=["P_PV_kW"], nrows=8760
)  # [kW] ([kWh/h])

# Creating a datetime type column
start_date = "2022-01-01 00:00:00"
start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
datetime_array = [
    start_datetime + datetime.timedelta(hours=i) for i in range(len(P_PV_data))
]

P_PV_data["Time"] = datetime_array
P_PV_data["P_PV_kW"] = P_PV_data["P_PV_kW"] * 5 # [kW]

# The electricity data refers to Sweden zone SE3 for year 2021
electricity_price_data = pd.read_csv(
    "PriceCurve_SE3_2021.csv", nrows=8760, header=0, sep=";"
)  # [cents/kWh]

# Random price chosen for hydrogen sale
hydrogen_price_data = 10  # [€/kg]


design = {
    "electroyzer_size": 1000,  # kW
    "solar_pv_size": 1800,  # kW
    "p_elec_max": 90,  # %
    "p_elec_min": 10,  # %
    "horizon": 1,  # year
    "deg_power": 0.5,  # %/year
    "sec_elec_nominal": 45,  # kWh/kgH2       #Specific Energy Consumption of the electrolyzer system as a whole
    "compressor_size": 1000,  # kW
    "compressor_pressure_out": 200,  # bar
    "compressor_pressure_in": 20,  # bar
    "compressor_sec_nominal": 0.1,  # kWh/kgH2
    "P_PV_data": P_PV_data,  # [kW]
    "Electricity_price_data": electricity_price_data,  # [cents/kWh]
    "Hydrogen_price_data": hydrogen_price_data,  # [€/kg]
    "Hydrogen_demand": 15,  # [kg/h]
    "V_sto_max" : 100, # %
    "V_sto_min" : 0, # %
}
