import pandas as pd
from sklearn.impute import SimpleImputer
df =pd.read_csv('framingham.csv')

#comprobando que cargue bien el dataframe

print(df.head())
print (df.info())
print(df.describe())

#Para el trabajo que voy a hacer, no encuentro necesario tener la columna 'education' tal vez esto podría estar relacionado con la
#probabildiad de ser fumador o consumir drogas. Pero esto directamente no se relaciona con la enfermedad cardiaca. Lo que sí se relaciona
#es el uso de sustancias, pero no su nivel educativo.

df = df.drop(columns=['education'])
print(df.info())

#vamos a buscar valores faltantes o nulos

print(df.isnull().sum())


#Tenemos 5 columnas con valores nulos, vamos a reemplazar esos datos con la media de cada columna. Primero creamos el imputador de datos.

meanimpputer = SimpleImputer(strategy='mean')

#Ahora aplicamos el imputador a las columnas con datos nulos

df[['cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']] = meanimpputer.fit_transform(df[['cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']])

#Comprobamos que ya no hay datos nulos

print (df.isnull().sum())


