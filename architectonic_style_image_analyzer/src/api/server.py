from flask import Flask, request, jsonify
import os, sys
import json
from PIL import Image
import io
import cv2
import numpy as np

UPLOAD_FOLDER = os.sep + "static" + os.sep
app = Flask(__name__)

ruta = os.path.abspath(__file__)
for i in range (3):
    ruta = os.path.dirname(ruta)

sys.path.append(ruta)

from src.utils.folders_tb import Gestor_archivos
from src.utils.models import Clasificador_ML
gestor = Gestor_archivos()
clasificador_ml = Clasificador_ML()


@app.route("/")
def home():
    """ Default path """
    return app.send_static_file('greet.html')


@app.route('/get_data', methods=['GET'])
def get_data():
    url_token = request.args.get('eltoken')
    token_file = UPLOAD_FOLDER + "token.json"
    json_readed = gestor.read_json(ruta + os.sep + 'src' + os.sep + 'api' + token_file)
    if url_token == json_readed['token']:
        json_dataset = gestor.read_json(ruta + os.sep + 'src' + os.sep + 'api' + os.sep + 'static' + os.sep + 'df_dataset.json')
        json_modelos = gestor.read_json(ruta + os.sep + 'src' + os.sep + 'api' + os.sep + 'static' + os.sep + 'df_modelos.json')
        json_funcion_mysql = gestor.read_json(ruta + os.sep + 'src' + os.sep + 'api' + os.sep + 'static' + os.sep + 'funcion_mysql.json')
        return json.dumps({'json_dataset': json_dataset, 'json_modelos': json_modelos, 'json_funcion_mysql': json_funcion_mysql})
    else:
        return 'La autenticación es incorrecta. Vuelve a intentarlo por favor.'



@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    response = {'success': False}
    if request.method == 'POST':
        if request.files.get('file'):
            img_requested = request.files['file'].read()
            img = Image.open(io.BytesIO(img_requested))
            cvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            model = gestor.carga_modelo_rnn(ruta + os.sep + 'models', 'modelo_rnn_12_data3.h5')
            pred, score = clasificador_ml.prediccion_rnn(model=model, img=cvImage, seed=42, knn=True, text=False)
            response['predictions'] = []
            row = {'label': pred, 'probability': float(score)} 
            response['predictions'].append(row)
            response['success'] = True
            return jsonify(response)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
    <p><input type=file name=file>
        <input type=submit value=Upload>
    </form>
    '''


def main():
    print("---------STARTING PROCESS---------")
    print(__file__)
    settings_file = UPLOAD_FOLDER + "settings.json"
    json_readed = gestor.read_json(ruta + os.sep + 'src' + os.sep + 'api' + settings_file)
    
    SERVER_RUNNING = json_readed["server_running"]
    print("SERVER_RUNNING", SERVER_RUNNING)
    if SERVER_RUNNING:
        DEBUG = json_readed["debug"]
        HOST = json_readed["host"]
        PORT_NUM = json_readed["port"]
        app.run(debug=DEBUG, host=HOST, port=PORT_NUM)
    else:
        print("Server settings.json doesn't allow to start server. " + 
            "Please, allow it to run it.")


main()
