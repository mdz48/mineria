import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

os.makedirs("img", exist_ok=True)

df = pd.read_csv("Walmart_Sales.csv")
print(df.head())
print(df.info())
print(df.describe())

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

plt.figure(figsize=(12, 7))
media_ventas = df['Weekly_Sales'].mean()
sns.histplot(df['Weekly_Sales'], kde=True, color='steelblue')
plt.axvline(media_ventas, color='red', linestyle='--', linewidth=2, label=f'Media: ${media_ventas:,.0f}')
plt.legend()
plt.tight_layout()
plt.savefig('img/fig1_hist_sales.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
df_box = df.copy()
df_box['Tipo Semana'] = df_box['Holiday_Flag'].map({0: 'Semana Normal', 1: 'Semana Festiva'})
sns.boxplot(data=df_box, x='Tipo Semana', y='Weekly_Sales')
plt.tight_layout()
plt.savefig('img/fig2_boxplot_holiday.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 7))
sns.scatterplot(data=df, x='Temperature', y='Weekly_Sales')
z = np.polyfit(df['Temperature'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
plt.plot(sorted(df['Temperature']), p(sorted(df['Temperature'])), 'r--', linewidth=2)
plt.tight_layout()
plt.savefig('img/fig3_scatter_temp.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.tight_layout()
plt.savefig('img/fig4_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    df['Temperature'],
    df['Fuel_Price'],
    df['Weekly_Sales']
)
plt.tight_layout()
plt.savefig('img/fig5_3d_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

fig6, ax6 = plt.subplots(figsize=(14, 7))
ventas_mensuales = df.groupby(['Year', 'Month'])['Weekly_Sales'].mean().reset_index()
ventas_mensuales['Date'] = pd.to_datetime(ventas_mensuales[['Year', 'Month']].assign(day=1))
ax6.plot(ventas_mensuales['Date'], ventas_mensuales['Weekly_Sales'], marker='o', linewidth=2, markersize=4, color='steelblue')
ax6.set_xlabel('Fecha', fontsize=12)
ax6.set_ylabel('Ventas Semanales Promedio (USD)', fontsize=12)
ax6.grid(True, alpha=0.3)
for year in [2010, 2011, 2012]:
    ax6.axvline(pd.Timestamp(f'{year}-12-01'), color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('img/fig6_time_series_monthly.png', dpi=150, bbox_inches='tight')
plt.close()
