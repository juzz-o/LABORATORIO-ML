import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models



df =pd.read_csv('framingham.csv')

#comprobando que cargue bien el dataframe.

print(df.head())
print (df.info())
print(df.describe())
print('------------------------------------------------------------------------------------------------------------------------')
#Para el trabajo que voy a hacer, no encuentro necesario tener la columna 'education' tal vez esto podría estar relacionado con la
#probabildiad de ser fumador o consumir drogas. Pero esto directamente no se relaciona con la enfermedad cardiaca. Lo que sí se relaciona
#es el uso de sustancias, pero no su nivel educativo.

df = df.drop(columns=['education'])
print(df.info())
print('------------------------------------------------------------------------------------------------------------------------')
#vamos a buscar valores faltantes o nulos.

print(df.isnull().sum())

print('------------------------------------------------------------------------------------------------------------------------')
#Tenemos 5 columnas con valores nulos, vamos a reemplazar esos datos con la media de cada columna. Primero creamos el imputador de datos.

meanimpputer = SimpleImputer(strategy='mean')

#Ahora aplicamos el imputador a las columnas con datos nulos.

df[['cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']] = meanimpputer.fit_transform(df[['cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']])

#Comprobamos que ya no hay datos nulos

print (df.isnull().sum())
print (df.info())

print('------------------------------------------------------------------------------------------------------------------------')





#Como ya tenemos el dataframe limpio, vamos a empezar a preparar los datos para el modelo de machine learning.
#Primero vamos a mirar cuáles columnas hay que estandarizar y cuáles no. (las que estén con valores de 1 y 0 no es necesario estandarizarlas)
#Y desde el principio vimos que nuestras características son todas numéricas, por lo que no es necesario hacer codificación one-hot o label encoding.
print('------------------------------------------------------------------------------------------------------------------------')
print(df.nunique()) 
print('------------------------------------------------------------------------------------------------------------------------')
#observamos que las columnas 'male', 'current smoker', 'prevalentStroke', 'prevalentHyp', 'diabetes' están en formato 1 y 0, por lo que no es necesario estandarizarlas.
#Aparte, nuestro target es la columna 'TenYearCHD' por lo que no hay que estandarizarla tampoco.
#Ahora crearemos un pipeline. Primero separamos las características del target.

X=df.drop(columns=['TenYearCHD'])
y=df['TenYearCHD']

#comenzamos con el pipeline definiendo las columnas numéricas y las binarias.

numerico = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
binario = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]

# creamos el pipeline para estandarizar solo las columnas numéricas.

scaler = Pipeline(steps=[
    ("scaler", StandardScaler())                   
])

#Separamos los datos en train, validation y test (70% train, 15% val, 15% test)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

#Ajustamos solo a los numéricos del train

X_train_num = scaler.fit_transform(X_train[numerico])
X_val_num   = scaler.transform(X_val[numerico])
X_test_num  = scaler.transform(X_test[numerico])

#Ahora juntamos los datos numéricos estandarizados con los datos binarios (que no necesitan estandarización)

X_train_final = np.hstack([X_train_num, X_train[binario].values])
X_val_final   = np.hstack([X_val_num, X_val[binario].values])
X_test_final  = np.hstack([X_test_num, X_test[binario].values])

#Entrenamos el modelo de KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
knn_val_acc = accuracy_score(y_val, knn.predict(X_val))
knn_test_acc = accuracy_score(y_test, knn.predict(X_test))

#hacemos el modelo de Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_val_acc = accuracy_score(y_val, rf.predict(X_val))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

#Hacemos el modelo de redes neuronales 

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30, batch_size=32, 
                    validation_data=(X_val, y_val), verbose=0)

dnn_train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
dnn_val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
dnn_test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

#Para ver los resultados en un dataframe hacemos lo siguiente

results = pd.DataFrame({
    "Modelo": ["kNN", "Random Forest", "Deep Neural Net"],
    "Train Accuracy": [knn_train_acc, rf_train_acc, dnn_train_acc],
    "Val Accuracy": [knn_val_acc, rf_val_acc, dnn_val_acc],
    "Test Accuracy": [knn_test_acc, rf_test_acc, dnn_test_acc]
})
print('------------------------------------------------------------------------------------------------------------------------')
print(results)
print('------------------------------------------------------------------------------------------------------------------------')
#probamos con un dato nuevo inventado


sample = pd.DataFrame([{
    'male': 1,              
    'age': 55,
    'currentSmoker': 1,
    'cigsPerDay': 10,
    'BPMeds': 0,
    'prevalentStroke': 0,
    'prevalentHyp': 1,
    'diabetes': 0,
    'totChol': 220,
    'sysBP': 140,
    'diaBP': 90,
    'BMI': 28.5,
    'heartRate': 78,
    'glucose': 110
}])

# Preprocesar la muestra

sample[numerico] = scaler.transform(sample[numerico])

sample_final = np.hstack([sample[numerico].values, sample[binario].values])
# Predecir con el modelo DNN

prob_dnn = model.predict(sample_final)  # entrega la probabilidad (entre 0 y 1)
pred_dnn = (prob_dnn > 0.5).astype(int)  # convertir a clase 0 o 1

print('------------------------------------------------------------------------------------------------------------------------')
print("Predicción (DNN):", pred_dnn[0][0])
print("Probabilidad de riesgo (DNN):", prob_dnn[0][0])
print('------------------------------------------------------------------------------------------------------------------------')
print(df['sysBP'].describe())

#ahora vamos a hacer un análisis exploratorio de datos para encontrar relaciones entre las características y el target y ver la distrubución de mis datos(TenYearCHD)
import matplotlib.pyplot as plt
import seaborn as sns
df.hist(figsize=(15, 10), bins=30)
plt.suptitle("Distribución de las variables", fontsize=16)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Matriz de correlación", fontsize=14)
plt.show()


plt.figure(figsize=(15, 10))

for i, col in enumerate(numerico, 1):
    plt.subplot(3, 3, i)  # rejilla de 3x3
    sns.boxplot(data=df, x="TenYearCHD", y=col, palette="Set2")
    plt.title(f"{col} vs TenYearCHD")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
corr_target = df.corr()["TenYearCHD"].drop("TenYearCHD").sort_values(ascending=False)
sns.barplot(x=corr_target, y=corr_target.index, palette="coolwarm")
plt.title("Correlación de variables con el riesgo (TenYearCHD)")
plt.show()



