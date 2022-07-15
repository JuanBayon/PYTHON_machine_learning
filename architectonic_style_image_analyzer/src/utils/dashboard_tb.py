import streamlit as st
from PIL import Image
import pandas as pd
import os, sys
from cv2 import imread
import cv2

ruta = os.path.abspath(__file__)
for i in range(3):
    ruta = os.path.dirname(ruta)

sys.path.append(ruta)

from src.utils.models import Clasificador_ML
from src.utils.folders_tb import Gestor_archivos
from src.utils.sql_tb import MySQL

gestor = Gestor_archivos()
clasificador_ml = Clasificador_ML()


configbd = gestor.read_json((ruta + os.sep + 'resources'+ os.sep + 'bd_info.json'))
IP_DNS = configbd["IP_DNS"]
USER = configbd["USER"]
PASSWORD = configbd["PASSWORD"]
BD_NAME = configbd["BD_NAME"]
PORT = configbd["PORT"]

mysql = MySQL(IP_DNS, USER, PASSWORD, BD_NAME, PORT)


class Gestor_streamlit:

    def configuracion(self):
        st.set_page_config(page_title='Clasificador de imágenes arquitectónicas', page_icon=':electric_plug:', layout="wide")


    def menu_home(self):
        img = Image.open(ruta + os.sep + 'resources' + os.sep + 'frame.jpg')
        st.image(img, use_column_width='auto')
        

        with st.beta_expander("DIFERENCIAR ESTILOS ARQUITECTÓNICOS DEL SIGLO XX"):
            st.write("""
            La arquitectura en el siglo XX dió un giro respecto a los años venideros y surgieron nuevos estilos más internacionales. 
            Para los que no están familiarizados con el tema es difícil saber qué estilo tiene alguno de los edificios más significativos.
            """)


        st.write(""" Se presenta un catalogador de imágenes de varios de estilos arquitectónicos más representativoas del siglo XX. Se pueden visualizar imágenes
        de ejemplos, así como repasar su historia. Quien esté interesado puede subir una imagen y se obtendrá la clasificación con su probabilidad de acierto""")



    def menu_historia(self):

        st.write('BREVE DESCRIPCIÓN DE LOS ESTILOS')

        st.write("""Se realiza una introducción a los estilos elegidos para este trabajo, todos ellos como se comentaba del siglo XX.""")

        st.subheader('ART NOUVEAU (MODERNISMO)')

        st.write("""- Corriente que se empezó a desarrolar a finales del sigo XIX hasta el siglo XX, (del 1890 al 1920). Un ejemplo en España muy representativo es Gaudí. 

                    - Representación de la llamada 'belle époque'

                    - Inspiración en la naturaleza incorporando los avances en materiales como el acero o el cristal. 

                    - Se propone democratizar la belleza y que hasta los objetos más cotidianos tuviese valor estético. 

                    - Uso de líneas curvas y sinuosas, con composiciones asimétricas. 

                    - Decoración y arquitectura se unen con formas orgánicas.""")


        st.subheader('ART DECO')

        st.write(""" - Movimiento popular entre 1920 y 1940 presente en todas las artes como moda, pintura, diseño, escultura o por supuesto arquitectura.

                    - El germen surgió en la exposición universal de 1900 de Paris con la formación de un colectivo dedicado a las artes decorativas 
                      de vanguardia o modernas como se autoidentificaban.

                    - Proviene del constructivismo, cubismo y futurismo. 

                    - Son líneas geometrizadas contundentes, muchas veces aerodinámicas en los bordes, como los aviones. Son composiciones simétricas.

                    - Con gran presencia de cubos y rascacielos con líneas sólidas.

                    - Con materiales como el aluminio, acero inoxidable o el cristal y  con especial cuidado en los ornamentos, con el uso de patrones 
                      y tipografías visibles.""")


        st.subheader('ESTILO INTERNACIONAL (RACIONALISMO)')

        st.write("""- Se desarrolló en todo el mundo como su nombre quiere indicar entre 1925 y 1965 como movimiento mayoritario en la primera mitad de siglo.

                    - Se identifica con los grandes maestros de la arquitectura como Mies van der Rohe o Le Corbusier.

                    - Arquitectura basada fundamentalmente en la razón de uso, con líneas sencillas y funcionales, basadas en formas geométricas 
                      simples (cubo y paralelepípedos, cilindro, esfera)

                    - Se usan materiales de tipo industrial (acero, hormigón, vidrio, etc)

                    - Las propuestas son para mejorar el uso que se hace de los espacios valiéndose de los nuevos materiales, sin nada superfluo. """)


        st.subheader('POSMODERNISMO')

        st.write("""- Respuesta al estilo internacional tan sobrio para volver al ingenio, al ornamento y a la referencia, que se desarrolla a partir de los años 50, establecido como movimiento en los 70. 

                    - Se usan ornamentos en fachada, con ángulos no ortogonales y superficies inusuales. Las cubiertas se dejan de hacer planas 
                      para volver a la forma del tejado. 

                    - Existe una intención de enfatizar los volúmenes y llamar la atención. 

                    - Los estilos se fusionan entre ellos, en el estilo que también es llamado 'neoecléctico' que mezcla diferentes estéticas.""")


        st.subheader('DECONSTRUCTIVISMO')

        st.write("""- Nace a finales de la década de los 80 y presente hasta hoy. 

                    - Se caracteriza por fragmentar y conseguir formas no rectilíneas ni planas. Se trata de una apariencia con un caos controlado. 

                    - Son edificios con multiples capas, con intención de una experimentación formal creando desequilibrios geométricos. 

                    - Un ejemplo muy reconocible en España es el Guggenheim de Bilbao.""")



    @st.cache(suppress_st_warning=True)
    def cargar_datos(self, csv_path):
        df = pd.read_csv(csv_path)
        return df



    def visualization(self):

        st.subheader("Se muestran varias imágenes ejemplo de cada uno de los estilos")

        st.subheader('ART NOUVEAU (MODERNISMO')
        img_nouveau = Image.open(ruta + os.sep + 'data' + os.sep + 'dataset_descargado' + os.sep + 'Art Nouveau' + os.sep + 'nouveau_13.jpg')
        st.image(img_nouveau, use_column_width='auto')
        st.write(
            
            """La casa batllo como es uno de los ejemplos más representativos de la arquitectura de Gaudí en España. 

        Una arquitecturabasada en la formas naturales, curvas y los elementos de decoración integrados""")


        st.subheader('ART DECO')
        img_deco = Image.open(ruta + os.sep + 'data' + os.sep + 'dataset_descargado' + os.sep + 'Art Deco' + os.sep + 'deco_4.jpg')
        st.image(img_deco, use_column_width='auto')
        st.write("""El edificio Chrysler en Nueva York es uno de los más representativos de este estilo.
        
        Está diseñado por el arquitecto William van Alen, con un interés claro por las artes decorativas y las texturas.""")

        st.subheader('ESTILO INTERNACIONAL (RACIONALIMSO')
        img_intern = Image.open(ruta + os.sep + 'data' + os.sep + 'dataset_descargado' + os.sep + 'Internacional' + os.sep + 'intern_1.jpg')
        st.image(img_intern, use_column_width='auto')
        st.write("""La casa Farnsworth es uno de los iconos de la arquitectura de estilo internacional.
        
        El proyecto es de Mies van der Rohe, autor de la frase 'Menos es Más'. Volumenes geométricos, color blanco y espacios basados en la utilidad. """)


        st.subheader('POSMODERNISMO')
        img_postmod = Image.open(ruta + os.sep + 'data' + os.sep + 'dataset_descargado' + os.sep + 'Posmoderno' + os.sep + 'postm_1.jpg')
        st.image(img_postmod, use_column_width='auto')
        st.write("""El edificio M2 de Kenzo Zuma en Tokio representa a la perfección este estilo.
        
        Tiene un estilo eclecticista y decorativo típico de los edificios postmodernistas""") 


        st.subheader('DECONSTRUCTIVISMO')
        img_decons= Image.open(ruta + os.sep + 'data' + os.sep + 'dataset_descargado' + os.sep + 'Deconstructivismo' +os.sep + 'decons_1.jpg')
        st.image(img_decons, use_column_width='auto')
        st.write("""El museo Guggenheim de Bilbao de Frank Gehry es un claro ejemplo de este estilo.
        
        Se basa en la destrucción y descomposición de los volúmenes creando multiples capas y envolventes.""")



    def file_selector(folder_path= ruta + os.sep + 'data' + os.sep + 'data_streamlit'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return os.path.join(folder_path, selected_filename)



    def model_prediction(self):

        st.sidebar.subheader('PRUEBA EL CLASIFICADOR CON ALGUNA IMAGEN')

        st.write("""El modelo utilizado para los resultados es una red neuronal preentrenada tipo RandomForest con las imagenes clusterizadas con
        el algoritmo de aprendizaje no supervisado knn a 25 colores""")

        modelo = gestor.carga_pickle(ruta + os.sep + 'models', 'modelo_forest_warm_start_2iter_data_origin.sav')
        parametros = str(["{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 3, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2000, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': True}"])
        modelo_sql = str(modelo)
        labels = ['Art Deco', 'Art Nouveau', 'Internacional', 'Posmoderno', 'Deconstructivismo']

        if st.checkbox('Seleccciona una imagen de las imágenes de muestra para el test'):
            folder_path = (ruta + os.sep + 'data' + os.sep + 'data_streamlit')
            if st.checkbox('Cambiar de carpeta'):
                folder_path = st.text_input('Introduce la ruta a la carpeta con la imagen', ruta + os.sep + 'data' + os.sep + 'data_streamlit')
            filenames = os.listdir(folder_path)
            selected_filename = st.selectbox('Select a file', filenames)
            pic = os.path.join(folder_path, selected_filename)
            st.write(f'Has seleccionado {selected_filename}')
            try:
                pic = cv2.imread(pic)
                st.image(pic, use_column_width='auto')
                pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
                imagen = cv2.resize(pic, (250, 250)).flatten() 
                imagen = imagen.reshape(-1, 187500)
                resultado = modelo.predict(imagen)
                probabilidad = modelo.predict_proba(imagen)
                sentencia = "Esta imagen pertenece al estilo arquitectónico {} con un {:.2f} % de confianza.".format(labels[resultado[0]], 100 * probabilidad[0][resultado[0]])
                st.write(sentencia)
                conn, engine = mysql.create_engine()
                sql = """INSERT INTO predictions (model, parameters, prediction) VALUES (%s, %s, %s)"""
                tuples = (modelo_sql, parametros, sentencia)
                mysql.execute_engine(conn, sql, tuples)
                mysql.close_engine(conn, engine)
            except: 
                pass
        
        elif st.checkbox('Selecciona una imagen de tu ordenador'):
            filename = st.text_input('Introduce la ruta de la imagen:')
            try:
                pic = cv2.imread(filename)
                st.image(pic, use_column_width='auto')
                pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
                imagen = cv2.resize(pic, (250, 250)).flatten() 
                imagen = imagen.reshape(-1, 187500)
                resultado = modelo.predict(imagen)
                probabilidad = modelo.predict_proba(imagen)
                sentencia = "Esta imagen pertenece al estilo arquitectónico {} con un {:.2f} % de confianza.".format(labels[resultado[0]], 100 * probabilidad[0][resultado[0]])
                st.write(sentencia)
                conn, engine = mysql.create_engine()
                sql = """INSERT INTO predictions (model, parameters, prediction) VALUES (%s, %s, %s)"""
                tuples = (modelo_sql, parametros, sentencia)
                mysql.execute_engine(conn, sql, tuples)
                mysql.close_engine(conn, engine)
            except:
                pass



    def predictions_historical(self):

        st.subheader('REVISA EL HISTORIAL DE PREDICCIONES HECHAS')

        st.write("""Se muestra el historial de predicciones hechas con el clasificador""")
        conn, engine = mysql.create_engine()
        data = pd.read_sql_table(table_name='predictions', con=conn)
        mysql.close_engine(conn, engine)

        st.dataframe(data)



    def json_api_flask(self):
        st.subheader('DESCARGA DE DATOS EN FORMATO JSON')
        st.write('Tienes a tu disposición los datos del proyecto utilizados para tu propio uso')
        st.write("""El archivo json disponible contiene 3 json:

                    json_dataset: contiene la lista de archivos y sus características usados para los modelos. 

                    json_modelos: contienen la lista de modelos entrenados con todas sus características para poder emularlos

                    json_funcion_mysq: contien el codigo de la función que se utilizó para guardar los dataframe en la base de datos mysql
        """)
        if st.button('Descargar'):
            download = pd.read_json('http://localhost:6060/get_data?eltoken=Q76903092')
            st.write('Tus datos se han descargado')
            
            st.subheader('TABLA CON EL LISTADO DE FOTOGRAFÍAS UTILIZADO')
            st.dataframe(download['json_dataset'])

            st.subheader('TABLA CON LA TABLA DE MODELOS PROBADOS Y SUS PARÁMETROS')
            st.dataframe(download['json_modelos'])

            st.subheader('TABLA CON LA FUNCIÓN UTILIZADA PARA GUARDAR LOS DATAFRAME EN LA BASE DE DATOS MYSQL')
            st.dataframe(download['json_funcion_mysql'])



    def ranking_models(self):

        st.subheader('REVISA LOS MODELOS PROBADOS Y SUS RESULTADOS')

        st.write("""Se muestra el dataframe con los diferentes modelos, variables utilizadas y métricas conseguidas""")
        conn, engine = mysql.create_engine()
        data = pd.read_sql_table(table_name='model_comparasion', con=conn)
        mysql.close_engine(conn, engine)

        st.dataframe(data.sort_values(by='score', ascending=False))