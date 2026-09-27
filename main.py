import pandas as pd
import numpy as np

# Загрузка данных
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(url)

# Q1. Pandas version
print("Q1:", pd.__version__)

# Q2. Records count 
print("Q2:", df.count)

# Q3. Fuel types 
print("Q3:", df['fuel_type'].unique())