from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, experimental
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import pickle
import numpy as np 
import os, sys
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV 
import random
from sklearn.cluster import KMeans
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.efficientnet import EfficientNetB2
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score
import cv2


ruta = os.path.abspath(__file__)
for i in range (3):
    ruta = os.path.dirname(ruta)

sys.path.append(ruta)

from src.utils.folders_tb import Gestor_archivos
gestor = Gestor_archivos()

from src.utils.mining_data_tb import Organizador
organizador = Organizador(['Art Deco', 'Art Nouveau', 'Internacional', 'Posmoderno', 'Deconstructivismo'], 250)



class Clasificador_ML():
    """
    
    """
    def __init__(self, size=250, batch_size=32):
        self.modelo_base = LogisticRegression()
        self.img_size = size
        self.batch_size = batch_size
        test_ds = organizador.get_data_tensor(path=(ruta + os.sep + 'data' + os.sep + 'dataset_modificado_3_test'), seed=42, validation_split=None, subset=None)
        self.labels = test_ds.class_names


    def train_cross_val(self, seed, model, x_train, y_train, splits, repeats=1, stratified=False, tree_warm_start=False):
        try: 
            if stratified:
                k_fold = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
        except ValueError:
            k_fold = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)

        val_score = []
        train_score = []
        train_index = []
        val_index = []

        for i, (train, val) in enumerate(k_fold.split(x_train)):
            print("Iteración:", i+1)
            print("val_size:", len(val))
            train_index.append(train)
            val_index.append(val)
        
            model.fit(x_train[train], y_train[train])
            if tree_warm_start:
                model.n_estimators += 100

            score_val = model.score(x_train[val], y_train[val])
            val_score.append(score_val)
            score_train = model.score(x_train[train], y_train[train])
            train_score.append(score_train)
            print('Score val:', score_val)
            print('Score train:', score_train)
            print('##########################')

        return train_score, val_score, train_index, val_index



    def entrena_modelo(self, model, seed, test_size=0.2, train_test=None, x=None, target=None, cv=None, fit=False):
        """
        train_test = (x_train, x_test, target_train, target_test)
        """
        if not train_test:
            x_train, x_test, target_train, target_test = train_test_split(x, target, test_size=test_size, random_state=seed)
        else:
            x_train, x_test, target_train, target_test = train_test

        scores_index = None
        if cv:
            scores_index = self.train_cross_val(seed, model, x_train, target_train, splits=cv[0], repeats=cv[1], stratified=cv[2], tree_warm_start=cv[3])
        elif not cv and fit:
            model.fit(x_train, target_train)

        if  not train_test:
            return x_train, x_test, target_train, target_test, scores_index
        else:
            return scores_index


    def grid_search(self, kind, X_train, y_train,):
        pipe = Pipeline(steps=[
            ('classifier', self.modelo_base)])

        if kind == 'logistic':
            params = {
                'classifier': [LogisticRegression()],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__C': [0.01, 0.1, 0.5, 1]}

        if kind == 'forest':
            params = {
                'classifier': [RandomForestClassifier()],
                'classifier__n_estimators': [10, 100, 1000],
                'classifier__max_features': [1,2,3]}

        if kind == 'svc':
            params = {
                'classifier': [svm.SVC()],
                'classifier__kernel': ('linear', 'rbf', 'sigmoid'),
                'classifier__C': [0.001, 1, 10],
                'classifier__gamma': ('scale', 'auto')}

        if kind == 'knn':
            params = {
                'classifier': [KNeighborsClassifier()],
                'classifier__n_neighbors': [3, 5, 11, 19],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ('euclidean', 'manhattan')}
            
        clf = GridSearchCV(estimator = pipe,
                        param_grid = params,
                        cv = 3,
                        verbose=1,
                        n_jobs=-1)

        clf.fit(X_train, y_train)
        return clf



    def resultados_grid(self, grid, train_test):
        print(grid.best_estimator_)
        print(grid.best_params_)
        print(grid.best_score_)
        self.indicativos_clasificacion(grid, train_test)



    def indicativos_clasificacion(self, model, train_test=None):

        x_train, x_test, target_train, target_test = train_test

        print('\n')
        print('PORCENTAJES----------------------------------------------------------')
        print('El score de acierto en el test es de:', model.score(x_test, target_test))
        print('El score de acierto en el entrenamiento es de:', model.score(x_train, target_train))

        print('\n')
        
        print('MATRIZ_CONFUSION TEST ------------------------------------------------')
        predictions_test = model.predict(x_test)
        print(confusion_matrix(target_test, predictions_test))
        print('\n')

        print('MATRIZ_CONFUSION ENTRENAMIENTO ---------------------------------------')
        predictions_train = model.predict(x_train)
        print(confusion_matrix(target_train, predictions_train))
        print('\n')

        predictions = predictions_test, predictions_train

        return predictions



    def genera_modelo(self, kind, seed, n_stimators=5, max_iter=900, penalty='l1', solver='lbfgs', kernel='rbf', gamma=0.1, C=1):
        if kind == 'logistic':
            model = LogisticRegression(max_iter=max_iter, penalty=penalty, solver=solver, n_jobs=-1, random_state=seed)
        elif kind == 'knn':
            model = KNeighborsClassifier(n_neighbors=n_stimators)
        elif kind == 'svc_linear':
            model = svm.LinearSVC(C=C, gamma=gamma, random_state=seed)
        elif kind == 'svc':
            model = svm.SVC(C=C, kernel=kernel, gamma=gamma, random_state=seed)
        elif kind == 'forest':
            model = RandomForestClassifier(n_estimators=n_stimators, random_state=seed)
        return model


    def genera_modelos_semillas(self, numero, kind, train_test, n_stimators=5, max_iter=900, penalty='l1', solver='lbfgs', kernel='rbf', gamma=0.1, C=1):
        semillas = []

        for i in range(0, numero):
            x = random.randint(1, 100000)
            semillas.append(x)

        modelos = []
        for semilla in semillas:
            model = self.genera_modelo(kind=kind, seed=semilla, n_stimators=n_stimators, max_iter=max_iter, penalty=penalty, solver=solver, kernel=kernel, gamma=gamma, C=C)
            result = self.entrena_modelo(model=model, seed=semilla, test_size=0.2, train_test=train_test, x=None, target=None, cv=None, fit=True)
            modelos.append(model)
        print('Las semillas elegidas para los modelos son:', semillas)

        return modelos


    def itera_max_iter_rl(self, max_iter, train_test, penalty, solver, seed):
        modelos = []

        for num_iter in max_iter:
            model = self.genera_modelo(kind='logistic', seed=seed, max_iter=num_iter, penalty=penalty, solver=solver)
            result = self.entrena_modelo(model=model, seed=seed, test_size=0.2, train_test=train_test, x=None, target=None, cv=None, fit=True)
            modelos.append(model)

        return modelos



    def resultado_modelos(self, modelos, train_test):

        x_train, x_test, y_train, y_test = train_test
        score_test = []
        score_train = []

        for model in modelos:
            sc_test = model.score(x_test, y_test)
            score_test.append(sc_test)
            sc_train = model.score(x_train, y_train)
            score_train.append(sc_train)
            print(f'Para el modelo {model} el modelo tiene estos resultados ##############################')
            print('El score de acierto en el test es de:', sc_test)
            print('El score de acierto en el entrenamiento es de:', sc_train)
            print('\n')

        indice_train = np.argmax(score_train)
        indice_test = np.argmax(score_test)

        print(f'El máximo score en el entrenamiento es {max(score_train)} con el modelo {modelos[indice_train]}')
        print(f'El máximo score en el entrenamiento es {max(score_test)} con el modelo {modelos[indice_test]}')



    def randomforest_cal_stimators(self, x, target, seed, path, train_test=None, n_stimators=100, max_features='auto', test_size=0.2, n_splits=10):

        if not train_test:
            X_train, X_test, y_train, y_test = train_test_split(x, target, test_size=test_size, random_state=seed)
            train_test = (X_train, X_test, y_train, y_test)
        else:
            X_train, X_test, y_train, y_test = train_test

        k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=seed)
        val_score = []
        train_score = []
        model = RandomForestClassifier(warm_start=True, n_estimators=n_stimators, max_features=max_features)

        for i, (train, val) in enumerate(k_fold.split(X_train)):
            print("Iteración:", i+1)
            print("val_size:", len(val))
        
            model.fit(X_train[train], y_train[train])
            model.n_estimators += 100

            score_val = model.score(X_train[val], y_train[val])
            val_score.append(score_val)
            score_train = model.score(X_train[train], y_train[train])
            train_score.append(score_train)
            print('Score val:', score_val)
            print('Score train:', score_train)
            print('##########################')

            if np.mean(val_score) > 0.99 and len(val_score) > 50:
                pickle.dump(model, open(path + "model_forest_warm_start", "wb"))
                print("STOP")
                break
            print('##########################')
            
        if not train_test:
            return model, train_test
        else:
            return model


    def red_neuronal_conv(self):
        model = Sequential()
        model.add(experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_size, self.img_size, 3)))
        model.add(Conv2D(32,3,padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128,activation="relu"))
        model.add(Dense(5, activation="softmax"))
        model.summary()
        return model



    def red_neuronal_conv_2(self):
        model = Sequential()
        model.add(experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_size, self.img_size, 3)))
        model.add(Conv2D(32, 3, padding="same", activation="tanh"))
        model.add(MaxPool2D())
        model.add(Conv2D(32, 3, padding="same", activation="tanh"))
        model.add(MaxPool2D())
        model.add(Conv2D(64, 3, padding="same", activation="tanh"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128,activation="tanh"))
        model.add(Dense(5, activation="softmax"))
        model.summary()
        return model



    def red_neuronal_3(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        kernel_initializer='he_normal',
                        input_shape=(self.img_size, self.img_size, 3)))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.summary()
        return model



    def red_neuronal_4(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        kernel_initializer='he_normal',
                        input_shape=(self.img_size, self.img_size, 3)))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(5, activation='softmax'))
        model.summary()
        return model



    def red_neuronal_preentrenada(self):
        base_model = VGG16(input_shape = (self.img_size, self.img_size, 3),
                        include_top=False,
                        weights = 'imagenet')

        for layer in base_model.layers:
            layer.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(5, activation='softmax')(x)

        model = tf.keras.models.Model(base_model.input, x)
        return model



    def rnn_preentrenada_2(self):
        base_model = EfficientNetB2(input_shape = (self.img_size, self.img_size, 3),
                        include_top=False,
                        weights = 'imagenet')

        for layer in base_model.layers:
            layer.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(5, activation='softmax')(x)

        model = tf.keras.models.Model(base_model.input, x)
        return model



    def entrena_modelo_rnn(self, model, train_ds, validation_data, epochs=100, optimizer='adam', early_stop=True):
        if model == 60:
            modelo = self.rnn_preentrenada_2()
        if model == 50:
            modelo = self.red_neuronal_preentrenada()
        elif model == 40:
            modelo = self.red_neuronal_4()
        elif model == 30:
            modelo = self.red_neuronal_3()
        elif model == 20:
            modelo = self.red_neuronal_conv_2()
        elif model == 10:
            modelo = self.red_neuronal_conv()

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

        if optimizer == 'adam':
            modelo.compile(optimizer = Adam(), 
                            loss = tf.keras.losses.CategoricalCrossentropy(), 
                            metrics = ['accuracy'])
        elif optimizer == 'sgd':
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            modelo.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                            optimizer=sgd, 
                            metrics = ['accuracy'])
        if early_stop:
            history = modelo.fit(train_ds, 
                                epochs=epochs, 
                                validation_data=validation_data, 
                                batch_size=self.batch_size, 
                                callbacks = [early_stop])
        else:
            history = modelo.fit(train_ds, 
                                epochs=epochs, 
                                validation_data=validation_data, 
                                batch_size=self.batch_size)
        return modelo, history



    def prediccion_rnn(self, model, img, seed, knn=False, text=True):
        if knn:
            X = img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=25, random_state=seed).fit(X)
            segmented_img = kmeans.cluster_centers_[kmeans.labels_]
            segmented_img = segmented_img.reshape(img.shape)
        else:
            segmented_img = img
        resized_pic = cv2.resize(segmented_img, (self.img_size, self.img_size))
        class_names = self.labels
        img_array = tf.keras.preprocessing.image.img_to_array(resized_pic)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        if text:
            resultado = "Esta imagen pertenece al estilo arquitectónico {} con un {:.2f} % de confianza.".format(class_names[np.argmax(score)], 100 * np.max(score))
            return resultado
        else:
            return class_names[np.argmax(score)], np.max(score)



    def resultado_test_rnn(self, modelo, test_ds):
        score = modelo.evaluate(test_ds, verbose=0)
        print('Los resultados para el conjunto de test son los siguientes')
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



    def resumen_modelos(self, path_modelos, path_test_list, seed):
        """
        lista_modelos = lista con los nombres de los modelos para cargarlos con picke.
        lista_rnn = lista de redes neuronales para cargar.
        """
        lista_modelos = [modelo for modelo in os.listdir(path_modelos) if not 'rnn' in modelo.split('_')]
        lista_rnn = [rnn for rnn in os.listdir(path_modelos) if 'rnn' in rnn.split('_') if 'history' not in rnn.split('_')]
        resumen = {'model': [], 'parameters': [], 'recall': [], 'score': []}

        for rnn in lista_rnn:
            model = gestor.carga_modelo_rnn(path_modelos, rnn)
            if 'origin' in rnn:
                test_ds = organizador.get_data_tensor(path=path_test_list[0], seed=seed, validation_split=None, subset=None, test=True)
            elif 'data1' in rnn:
                test_ds = organizador.get_data_tensor(path=path_test_list[1], seed=seed, validation_split=None, subset=None, test=True)
            elif 'data2' in rnn:
                test_ds = organizador.get_data_tensor(path=path_test_list[2], seed=seed, validation_split=None, subset=None, test=True)
            elif 'data3' in rnn:
                test_ds = organizador.get_data_tensor(path=path_test_list[3], seed=seed, validation_split=None, subset=None, test=True)

            resumen['model'].append(f'{model}')
            config = model.get_config()
            resumen['parameters'].append(config)
            evaluate_test = model.evaluate(test_ds, verbose=0)
            resumen['score'].append(evaluate_test[1])
            y = np.concatenate([y for x, y in test_ds], axis=0)
            array_y = np.argmax(y, axis=1)
            prediction = model.predict(test_ds)
            array_prediction = np.argmax(prediction, axis=1)
            recall = recall_score(array_y, array_prediction, average='micro')
            resumen['recall'].append(recall)

        for modelo in lista_modelos:
            if 'origin' in modelo:
                x_test, y_test = organizador.get_data(data_dir=path_test_list[0], flatten=True, color=True)
            elif 'data1' in modelo:
                x_test, y_test = organizador.get_data(data_dir=path_test_list[1], flatten=True, color=True)
            elif 'data2' in modelo:
                x_test, y_test = organizador.get_data(data_dir=path_test_list[2], flatten=True, color=True)
            elif 'data3' in modelo:
                x_test, y_test = organizador.get_data(data_dir=path_test_list[3], flatten=True, color=True)
            model = gestor.carga_pickle(path_modelos, modelo)
            if isinstance(model, list):
                for modelo_semilla in model:
                    resumen['model'].append(f'{modelo_semilla}')
                    resumen['model'].append(f'{modelo_semilla}')
                    params = modelo_semilla.get_params()
                    resumen['parameters'].append(params)
                    predictions_test = modelo_semilla.predict(x_test)
                    score = accuracy_score(y_test, predictions_test)
                    resumen['score'].append(score)
                    recall = recall_score(y_test, predictions_test, average='micro')
                    resumen['recall'].append(recall)
                    
            else:
                resumen['model'].append(f'{model}')
                params = model.get_params()
                resumen['parameters'].append(params)
                predictions_test = model.predict(x_test)
                score = accuracy_score(y_test, predictions_test)
                resumen['score'].append(score)
                recall = recall_score(y_test, predictions_test, average='micro')
                resumen['recall'].append(recall)
        
        return pd.DataFrame(resumen)


    def guarda_capas_modelo(self, modelo):
        layers = {}
        for i in range(len(modelo.layers)):
            weightsAndBiases = modelo.layers[i].get_weights
            layers[f'layer_{i}'] = weightsAndBiases
        return layers
