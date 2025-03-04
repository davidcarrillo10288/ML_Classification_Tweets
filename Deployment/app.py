import pickle
from convertir_texto import convertir_texto
from convertir_texto import tranformation
import streamlit as st
import joblib
import requests
import io
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

st.title('Modelo de Clasificación de Tweets')
st.subheader('Aplicación del modelo de clasificación de tweets')

## cargar modelos entrenados en colab
@st.cache_resource
def cargar_modelo():
    # URL del archivo en GitHub (en formato "raw")
    url = "https://github.com/davidcarrillo10288/ML_Classification_Tweets/raw/master/modelos/modelo_xgb_new.pkl"
    
    # Descargar el archivo
    response = requests.get(url)
    response.raise_for_status()  # Verifica que la descarga fue exitosa
    
    # Cargar el modelo desde el contenido descargado
    modelo = joblib.load(io.BytesIO(response.content))
    return modelo

@st.cache_resource
def cargar_scaler():
    # URL del archivo en GitHub (en formato "raw")
    url = "https://github.com/davidcarrillo10288/ML_Classification_Tweets/raw/master/modelos/scaler_xgb.pkl"
    
    # Descargar el archivo
    response = requests.get(url)
    response.raise_for_status()  # Verifica que la descarga fue exitosa
    
    # Cargar el modelo desde el contenido descargado
    modelo = joblib.load(io.BytesIO(response.content))
    return modelo

# Cargar los modelos

modelo_cargado = cargar_modelo()
scaler_cargado = cargar_scaler()

def prediction(df_num):
  df_num = scaler_cargado.transform(df_num)
  prediccion = modelo_cargado.predict(df_num)
  if prediccion == 0:
    result = 'Negativo'
  else:
    result = 'Positivo'
  return result

if __name__ == '__main__':
  text = st.text_input("Ingrese un texto:")
  if text:
      df = convertir_texto(text)
      df_num = tranformation(convertir_texto(df))
      result = prediction(df_num)
      st.write(f'El texto es {result}')