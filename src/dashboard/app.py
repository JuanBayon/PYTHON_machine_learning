import streamlit as st
import os, sys

ruta = os.path.abspath(__file__)
for i in range (3):
    ruta = os.path.dirname(ruta)

sys.path.append(ruta)

from src.utils.dashboard_tb import Gestor_streamlit

gestor = Gestor_streamlit()

csv_path = ruta + os.sep + 'data' + os.sep +'dataframe_resumen_final.csv'

gestor.configuracion()
df = gestor.cargar_datos(csv_path)


menu = st.sidebar.selectbox('Menu:',
                    options=['Bienvenida', 'Historia', 'Visualización', 'Descarga Json(API-Flask)', 'Predicción con el modelo', 'Modelos utilizados(SQL)', 'Histórico de predicciones(SQL)'])

st.title('Clasificador de imágenes de estilos arquitectónicos del siglo XX')

if menu == 'Bienvenida':
    gestor.menu_home()
elif menu == 'Historia':
    gestor.menu_historia()
elif menu == 'Visualización':
    gestor.visualization()
elif menu == 'Descarga Json(API-Flask)':
    gestor.json_api_flask()
elif menu == 'Predicción con el modelo':
    gestor.model_prediction()
elif menu == 'Modelos utilizados(SQL)':
    gestor.ranking_models()
elif menu == 'Histórico de predicciones(SQL)':
    gestor.predictions_historical()