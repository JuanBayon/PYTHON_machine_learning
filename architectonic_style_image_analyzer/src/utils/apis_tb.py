import json
import argparse


class Gestor_json:

    def read_json(self, fullpath):
        """
        The function reads a json given the full path with the file
        at the end and returns a variable wiht the dictionary
        """
        with open(fullpath, "r") as json_file_readed:
            json_readed = json.load(json_file_readed)
        return json_readed



    def load_json(self, name, fullpath):
        """
        The function writes a json file given a dictironary and
        the full path where you want to save it. 
        """
        with open(fullpath, 'w+') as outfile:
            json.dump(name, outfile, indent=4)



class Parser():

    def __init__(self):
        self.parser = self.crear_parser()

    def crear_parser(self):
        """
        Crea un parser para definir argumentos a pasar por
        consola al ejecutar el programa y lo retorna.
        """
        parser = argparse.ArgumentParser()
        return parser


    def agregar_argumento(self, numero, tipo:list):
        """
        Añade el número de argumentos que se quiera, pasado además el parser
        y la lista con los tipos de cada argumento.
        """
        for i in range(numero):
            self.parser.add_argument("-x", "--x", type=tipo[i], help="Password")


    def recoger_argumentos(self):
        """
        Recoge los argumentos pasados por consola al ejecutar
        el programa y los retorna.
        """
        args = vars(self.parser.parse_args())
        return args