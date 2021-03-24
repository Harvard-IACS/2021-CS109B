# Importing libraries
import numpy as np
import tensorflow as tf
from numpy.random import seed

seed(1)
tf.random.set_seed(1)

from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img
from PIL import Image


# Define a function to plot the train and validation accuracy and loss

def plot_history(history, name):
    with plt.xkcd(scale=0.2):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for i, metric in enumerate(['loss', 'accuracy']):
            ax[i].plot(history.history[metric], label='Train', color='#EFAEA4', linewidth=3)
            ax[i].plot(history.history[f'val_{metric}'], label='Validation', color='#B2D7D0', linewidth=3)
            if metric == 'accuracy':
                ax[i].axhline(0.5, color='#8d021f', ls='--', label='Trivial accuracy')
                ax[i].set_ylabel("Accuracy", fontsize=14)
            else:
                ax[i].set_ylabel("Loss", fontsize=14)
            ax[i].set_xlabel('Epoch', fontsize=14)

        plt.suptitle(f'{name} Training', y=1.05, fontsize=16)
        plt.legend(loc='best')
        plt.tight_layout()
