import os 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import seaborn as sns

df = pd.read_csv("assets/plant_power_2026-05-29_2026-05-29.csv")
print(df.head())
print(df.info())
print(df.describe())

duplicados = df.duplicated().sum()
print('Cantidad de registros duplicados de ambas columnas:', duplicados)

# Boxplot
plt.figure(figsize=(6, 6))
plt.boxplot(df['power_w'])
plt.title('Boxplot de power_w')
plt.xlabel('Potencia (W)')
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(df['power_w'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribucion de power_w')
plt.xlabel('Potencia (W)')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Grafica del tiempo contra si misma
plt.figure(figsize=(10, 5))
lag_plot(df['timestamp_utc'])
plt.title('Autocorrelacion de timestamp_utc')
plt.show()

# Grafica correlacional
plt.figure(figsize=(10, 5))
plt.scatter(df['timestamp_utc'], df['power_w'])
plt.title('Correlacion entre timestamp_utc y power_w')
plt.xlabel('timestamp_utc')
plt.ylabel('power_w')
plt.show()

# heatmap
sns.heatmap(df['timestamp_utc'], df['power_w'])
plt.title('Correlacion entre timestamp_utc y power_w')
plt.xlabel('timestamp_utc')
plt.ylabel('power_w')
plt.show()
