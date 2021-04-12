# Import necessary libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import seed
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K


def get_data():
	# Set random seed
	seed(1)
	tf.random.set_seed(1)

	# Read the MNIST dataset 
	mnist = tf.keras.datasets.mnist

	# Split the data into train and test sets
	# (x_train,y_train),(x_test,y_test)= mnist.load_data()
	_, (x, y) = mnist.load_data(path="mnist.npz")

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)

	# Reshaping the array to 4-dims so that it can work with the Keras API
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	input_shape = (28, 28, 1)

	# Ensuring that the values are float so that we can get decimal points after division
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	# Normalizing the RGB codes by dividing it to the max RGB value.
	x_train /= 255
	x_test /= 255

	return x_train, y_train, x_test, y_test


# Function to plot the activation of the first layer
def plot_activation(model, x_train):
	inp = model.input
	outputs = [layer.output for layer in model.layers]
	functors = [K.function([inp], [out]) for out in outputs]

	layer_outs = [func([x_train[0:1]]) for func in functors]
	last = layer_outs[0]

	f, axs = plt.subplots(1, 2, figsize=(15, 15))

	axs[0].imshow(x_train[0].squeeze(), cmap="bone")
	axs[0].set_title("Train data", fontsize=20)

	axs[1].imshow(last[0][0][:, :, -1].squeeze(), cmap="bone")
	axs[1].set_title("Activation", fontsize=20);
