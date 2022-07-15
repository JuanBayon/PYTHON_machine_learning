import os 
import cv2
import numpy as np 
import tensorflow as tf
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from matplotlib.image import imread
from sklearn.cluster import KMeans
from ast import literal_eval
import pandas as pd 

class Organizador():
    
    
    def __init__(self, labels, img_size):
        self.labels = labels
        self.img_size = img_size


    def dataframe_resumen_dataset(self, data_dir):
        data = {'filename': [], 'metadata': []}
        for label in self.labels: 
            path = os.path.join(data_dir, label)
            class_num = self.labels.index(label)
            for img in os.listdir(path):
                try:
                    pic = cv2.imread(os.path.join(path, img))
                    size = pic.shape
                    data['filename'].append(str(img))
                    metadata = {'img_size': size, 'img_label': class_num}
                    data['metadata'].append(metadata)
                except Exception:
                    continue
        
        return pd.DataFrame(data, columns=['filename', 'metadata'])


    def get_data(self, data_dir, flatten=True, color=True, muestra=None):
        data = []
        for label in self.labels: 
            path = os.path.join(data_dir, label)
            class_num = self.labels.index(label)
            for img in os.listdir(path):
                try:
                    pic = cv2.imread(os.path.join(path, img))
                    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
                    if flatten:
                        resized_pic = cv2.resize(pic, (self.img_size, self.img_size)).flatten() 
                    else:
                        resized_pic = cv2.resize(pic, (self.img_size, self.img_size))
                    data.append([resized_pic, class_num])
                except Exception:
                    continue

        random.shuffle(data)
        x_train = []
        y_train = []
        if muestra:
            for x, y in data[:muestra]:
                x_train.append(x)
                y_train.append(y)
        else:
            for x, y in data:
                x_train.append(x)
                y_train.append(y)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train =  x_train.astype("float32") / 255.0
        if color:
            x_train.reshape(-1, self.img_size, self.img_size, 3)
        else:
            x_train.reshape(-1, self.img_size, self.img_size, 1)

        return x_train, y_train


    def carga_imagenes_etiqueta(self, data_dir, label):

        images = []
        path = os.path.join(data_dir, label)

        for img in os.listdir(path):
            try:
                pic = cv2.imread(os.path.join(path, img))
                resized_pic = cv2.resize(pic, (self.img_size, self.img_size))
                images.append(resized_pic)
            except Exception:
                continue

        random.shuffle(images)
        images = np.array(images)
        return images


    def get_data_tensor(self, path, seed, validation_split=None, subset=None, batch_size=32, test=False):
        if test: 
            data = tf.keras.preprocessing.image_dataset_from_directory(
            directory=path, labels='inferred', label_mode='categorical',
            class_names=None, color_mode='rgb', batch_size=10000, image_size=(self.img_size,
            self.img_size), shuffle=False, seed=seed, validation_split=validation_split, subset=subset,
            interpolation='bilinear', follow_links=False, smart_resize=False)
            return data
        else:
            data = tf.keras.preprocessing.image_dataset_from_directory(
            directory=path, labels='inferred', label_mode='categorical',
            class_names=None, color_mode='rgb', batch_size=batch_size, image_size=(self.img_size,
            self.img_size), shuffle=True, seed=seed, validation_split=validation_split, subset=subset,
            interpolation='bilinear', follow_links=False, smart_resize=False)
            return data


    def train_test(self, root_dir, label, test_ratio=0.05):

        src = root_dir + os.sep + label

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)* (1 - test_ratio))])

        train_FileNames = [src+ os.sep + name for name in train_FileNames.tolist()]
        test_FileNames = [src+ os.sep + name for name in test_FileNames.tolist()]

        print(f'Test_Train for label {label} -------------')
        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Testing: ', len(test_FileNames))
        print('\n')

        for name in train_FileNames:
            shutil.copy(name, root_dir +'_train' + os.sep + label)

        for name in test_FileNames:
            shutil.copy(name, root_dir +'_test' + os.sep + label)


    def separa_train_test(self, path, test_ratio=0.05):
        
        for label in self.labels:
            os.makedirs(path + '_train' + os.sep + label)
            os.makedirs(path + '_test' + os.sep + label)
            self.train_test(path, label, test_ratio)


    def make_copy_label(self, root_dir, label, etiqueta, root_destiny):

        src = root_dir + os.sep + label

        allFiles = [src + os.sep + f'{etiqueta}{i}.jpg' for i in range(1, 780)]
        try:
            for name in allFiles:
                shutil.copy(name, root_destiny + os.sep + label)
        except Exception:
            pass


    def make_copy_dataset(self, root_dir, label, etiquetas, root_destiny):
        for etiqueta, label in zip(etiquetas, self.labels):
            os.makedirs(root_destiny + os.sep + label)
            self.make_copy_label(root_dir, label, etiqueta, root_destiny)



    def generator(self, x_train):

        datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range = 30, 
        zoom_range = 0.2, 
        width_shift_range=0.1,  
        height_shift_range=0.1, 
        horizontal_flip = True,
        fill_mode="nearest",
        shear_range=0.15,
        vertical_flip=False)
        datagen.fit(x_train)

        return datagen


    def genera_nuevos(self, images, save_path, save_prefix, number, color=True):

        if color:
            images.reshape(-1, self.img_size, self.img_size, 3)
        else:
            images.reshape(-1, self.img_size, self.img_size, 1)
        datagen = self.generator(images)
        imageGen = datagen.flow(images, batch_size=1, save_to_dir=save_path, save_prefix=save_prefix, save_format="jpg")
        total = 0
        for image in imageGen:
            total += 1
            if total == number:
                break


    def configure_for_performance(self, ds):
        AUTOTUNE = tf.data.AUTOTUNE
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds


    def no_supervisado_kmeans_rebaja_colores(self, ruta, etiquetas, seed, makedirs=True , ruta_origen=None, ruta_destino=None):
        if not ruta_origen:
            ruta_origen = ruta + os.sep + 'data' + os.sep + 'dataset_original'
        if not ruta_destino:
            ruta_destino = ruta + os.sep + 'data' + os.sep + 'dataset_modificado'
        for etiqueta, label in zip(etiquetas, self.labels):
            if makedirs == True:
                os.makedirs(ruta_destino + os.sep + label)
            for i in range(1, 780):
                try:
                    image = imread(ruta_origen + os.sep + label + os.sep + f'{etiqueta}{i}.jpg')
                    X = image.reshape(-1, 3)
                    kmeans = KMeans(n_clusters=25, random_state=seed).fit(X)
                    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
                    segmented_img = segmented_img.reshape(image.shape)
                    cv2.imwrite(ruta_destino + os.sep + label + os.sep + f'{etiqueta}knn_{i}.jpg', segmented_img)
                except Exception:
                    continue
            print(f'Las fotografías de la etiqueta {label} se han bajado a 25 colores con knn')






class Organizador_dataframe():


    def simplifica_columna_parametros_tensorflow(self, df, filas):
    
        df_mysql = df.copy()
        for fila in filas:
            if fila == 0:
                capas = ['EfficientNetB2', 'Dense', 'Dense'] 
            else:
                diccionario = literal_eval(df_mysql.loc[fila, 'parameters'])
                capas = []
                for capa in diccionario['layers']:
                    capas.append(capa['class_name'])

            df_mysql.at[fila, 'parameters'] = str(capas)
        
        return df_mysql
