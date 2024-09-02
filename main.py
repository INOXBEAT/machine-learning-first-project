import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Cargar el conjunto de datos de viviendas en California
cali_data = fetch_california_housing()
df = pd.DataFrame(cali_data.data, columns=cali_data.feature_names)
df['MedHouseVal'] = cali_data.target

# Exploración de los datos
print(df.head())
print(df.info())
print(df.describe())

# Correlación entre características
correlation = df.corr()
print(correlation['MedHouseVal'].sort_values(ascending=False))

# Visualización de la relación entre una característica y el valor medio de la vivienda
plt.scatter(df['AveRooms'], df['MedHouseVal'])
plt.xlabel('Número medio de habitaciones')
plt.ylabel('Valor medio de la vivienda')
plt.title('Relación entre número medio de habitaciones y valor medio de la vivienda')
plt.show()
