import pandas as pd
import numpy as np

# Загрузка данных
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(url)

# Q1. Pandas version
print("Q1:", pd.__version__)

# Q2. Records count 
print("Q2:", df.shape[0])

# Q3. Fuel types 
print("Q3:", df['fuel_type'].unique())

# Q4. Missing values
print("Q4:", len(df.isnull().sum()[df.isnull().sum() > 0]))

# Q5. Max fuel efficiency
asia = df[df['origin'] == 'Asia']
print("Q5:", asia['fuel_efficiency_mpg'].max())

# Q6.  Median value of horsepower
median_before = df['horsepower'].median()
most_frequent = df['horsepower'].mode()[0]
df['horsepower'] = df['horsepower'].fillna(most_frequent)
median_after = df['horsepower'].median()

if median_after > median_before:
    q6 = "Yes, it increased"
elif median_after < median_before:
    q6 = "Yes, it decreased"
else:
    q6 = "No"
print("Q6:", q6)
print(f"    median before: {median_before}, after: {median_after}")

# Q7. Sum of weights
asia = df[df['origin'] == 'Asia']
X_df = asia[['vehicle_weight', 'model_year']].head(7)
X = X_df.values

XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y

print("Q7:", w.sum())