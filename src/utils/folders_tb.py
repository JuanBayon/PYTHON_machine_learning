import os, sys
import pandas as pd 
import pickle
import tensorflow as tf 

ruta = os.path.abspath(__file__)
for i in range (3):
    ruta = os.path.dirname(ruta)

sys.path.append(ruta)

from src.utils.apis_tb import Gestor_json

class Gestor_archivos(Gestor_json):
    
    def cargar_df(self, ruta):
            return pd.read_csv(ruta)
    

    def guardar_csv(self, df, ruta, nombre):
        df.to_csv(ruta + os.sep + 'data' + os.sep + nombre, index=False)


    def guarda_pickle(self, variable, path, nombre):
        full_path = path + os.sep + nombre
        pickle.dump( variable, open(full_path, "wb" ))


    def carga_pickle(self, path, nombre):
        full_path = path + os.sep + nombre
        return pickle.load(open(full_path, "rb" ))


    def guarda_history_rnn(self, history, path, nombre):
        full_path = path + os.sep + nombre
        with open(full_path, 'wb') as file:
            pickle.dump(history.history, file)


    def carga_modelo_rnn(self, path, nombre):
        full_path = path + os.sep + nombre
        loaded_model = tf.keras.models.load_model(full_path)
        return loaded_model
