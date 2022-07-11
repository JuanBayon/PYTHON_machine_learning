import os
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np


class Gestor_apariencia:

    def set_theme(self, option):
        plt.style.use('seaborn')
        if option == 'a':
            sns.set(rc={'axes.facecolor':'cadetblue', 
                        'figure.facecolor':'cadetblue', 
                        'grid.color': 'turquoise', 
                        'grid.linestyle': ':', 
                        'xtick.color': 'white', 
                        'ytick.color': 'white', 
                        'text.color': 'white', 
                        'axes.labelsize': 10, 
                        'axes.titlesize': 15, 
                        'font.size':10,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10})
        if option == 'b':
            sns.set(rc={'axes.facecolor':'indianred', 
                        'figure.facecolor':'indianred', 
                        'grid.color': 'lightpink', 
                        'grid.linestyle': ':', 
                        'xtick.color': 'white', 
                        'ytick.color': 'white', 
                        'text.color': 'white', 
                        'axes.labelsize': 10, 
                        'axes.titlesize': 15, 
                        'font.size':10,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10})
        if option == 'c':
            sns.set(rc={'axes.facecolor':'steelblue', 
                        'figure.facecolor':'steelblue', 
                        'grid.color': 'aliceblue', 
                        'grid.linestyle': ':', 
                        'xtick.color': 'white', 
                        'ytick.color': 'white', 
                        'text.color': 'white', 
                        'axes.labelsize': 10, 
                        'axes.titlesize': 15, 
                        'font.size':10,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10})



class Plotter(Gestor_apariencia):

    def default_values(self, labelx=None, labely=None, title=None, rotation=None, figsize=(6, 6), ylim=None):
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylabel(labely, color='white', labelpad=20)
        plt.xlabel(labelx, color='white', labelpad=20)
        plt.title(title, pad=35)
        plt.xticks(size=15, rotation=rotation)
        plt.yticks(size=15)
        sns.despine(right=False, top=False)
        if ylim:
            ax.set_ylim(ylim)
        return fig, ax


    def visualizar(self, kind, df=None, x=None, y=None, ax=None, bins=None, hue=None, external_xlabel=None, external_ylabel=None, color='white', label=None):
        if kind == 'lm':
            h = sns.lmplot(x=x, y=y, data=df, line_kws={'color': color}, scatter_kws={'color': color}, x_bins=bins, hue=hue)
            h.set_axis_labels(external_xlabel, external_ylabel, color=color)
            sns.despine(right=False, top=False)
            return h
        elif kind == 'line':
            sns.lineplot(data=df, x=x, y=y, color=color, ax=ax, hue=hue)
        elif kind == 'joint':
            h = sns.jointplot(x=x, y=y, data=df, color=color, ax=ax, hue=hue)
            h.set_axis_labels(external_xlabel, external_ylabel, color=color)
            return h
        elif kind == 'hist':
            if not bins:
                sns.histplot(x=x, data=df, color=color, ax=ax, hue=hue, label=label)
                if label:
                    ax.legend(prop={'size': 12}) 
            else: 
                sns.histplot(x=x, data=df, color=color, ax=ax, bins=bins, hue=hue, label=label)
                if label:
                    ax.legend(prop={'size': 12}) 
        elif kind == 'count':
            h = sns.countplot(x=x, data=df, color=color, alpha=.8, ax=ax, hue=hue)
        elif kind == 'bar':
            h = sns.barplot(x=x, y=y, data=df, color=color, alpha=.8, ax=ax, hue=hue)
            h.set(xlabel=external_xlabel, ylabel=external_ylabel)
        elif kind == 'dist':
            sns.distplot(x=x, color=color, ax=ax, bins=bins)
        elif kind == 'cat':
            sns.catplot(x=x, y=y, data=df, color=color, ax=ax, hue=hue)
        elif kind == 'scatter':
            sns.scatterplot(x=x, y=y, data=df, color=color, ax=ax, hue=hue)
        elif kind == 'dis':
            sns.displot(x=x, y=y, data=df, color=color, ax=ax, hue=hue)
        elif kind == 'box':
            # self.set_boxplot_lines_colours(ax)
            sns.boxplot(x=x, data=df, color=color, flierprops={'markerfacecolor':'dimgray'} , ax=ax, hue=hue)
            plt.setp(ax.lines, color='dimgray')


    def set_boxplot_lines_colours(self, ax):
        for i in range(6):
            line = ax.lines[i]
            line.set_color('white')
            line.set_mfc('white')
            line.set_mec('white')


    def guardar_figura(self, fig, nombre, ruta):
        fig.savefig(ruta + os.sep + 'reports' + os.sep + nombre)


    def pie_chart(self, df, titulo):
        fig, ax = plt.subplots(figsize=(9, 9))
        cmap = plt.get_cmap('Set3')
        colors = [cmap(i) for i in np.linspace(0, 1, 8)]
        patches, texts, pcts = ax.pie(df.values, labels=df.index, autopct='%1.1f%%', colors=colors, textprops={'size': 'x-large'})
        plt.setp(pcts, color='dimgray', fontweight='bold')
        ax.set_title(titulo, fontsize=18)
        return fig


    def representa_accuracy_loss_rnn(self, history, epochs_range, guardada=True):

        if guardada:
            acc = history['accuracy']
            val_acc = history['val_accuracy']
            loss = history['loss']
            val_loss = history['val_loss']
        else:
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

        epochs_range = range(epochs_range)

        fig = plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, acc, label='Acierto en el entrenamiento')
        plt.plot(epochs_range, val_acc, label='Acierto en la validación')
        plt.legend(loc='lower right')
        plt.title('Acierto en el entrenamiento y la validación')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='Error en el entrenamiento')
        plt.plot(epochs_range, val_loss, label='Error en la validación')
        plt.legend(loc='upper right')
        plt.title('Error en el entrenamiento y la validación')
        plt.show()
        return fig



    def visualiza_fotos(self, train_ds):
        image_batch, label_batch = next(iter(train_ds))
        class_names = train_ds.class_names
        fig = plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 14, 'text.color': 'steelblue', 'font.weight': 'bold'})
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch[i]
            label = np.argmax(label)
            plt.title(class_names[label])
            plt.axis("off")
        return fig
