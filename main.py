import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# Separar las características (X) de la variable objetivo (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")

# Crear una instancia del modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train, y_train)

# Evaluar el modelo con el conjunto de prueba
score = model.score(X_test, y_test)
print(f"Puntuación del modelo en el conjunto de prueba: {score}")
    