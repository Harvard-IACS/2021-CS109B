import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input


def load_dataset():
	df = pd.read_csv("cifar.csv")
	print(df.shape)
	generator = ImageDataGenerator(rescale=1. / 255)

	data_gen = generator.flow_from_dataframe(
		df, directory=None, x_col='image', y_col='label',
		target_size=(32, 32), color_mode='rgb', seed=30,
		class_mode='categorical', batch_size=50, shuffle=False,
		save_format='png', subset='training')

	return data_gen, df


# Create function to apply a grey patch on an image
def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
	patched_image = np.array(image, copy=True)
	patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = 127.5
	return patched_image


def occlusion(model, img_num=10, patch_size=4):
	# Load image from the test data
	data_gen, df = load_dataset()
	cifar_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
				  5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

	num = img_num

	if num > 49:
		num = 1

	# Get image 
	img = data_gen[0][0][num]

	# Get label
	y_test = data_gen[0][-1]

	# Get the patch size for occlusion
	PATCH_SIZE = patch_size

	# Get the loss of the model with the original image
	loss = model.evaluate(img.reshape(-1, 32, 32, 3), y_test[num].reshape(-1, 10), verbose=0)

	# Define a numpy array to store the loss differences
	loss_map = np.zeros((img.shape[0], img.shape[1]))

	# Iterate the patch over the entire image
	for top_left_x in range(0, img.shape[0], PATCH_SIZE):

		for top_left_y in range(0, img.shape[1], PATCH_SIZE):
			# Initialise a new patched image
			patched_image = apply_grey_patch(img, top_left_x, top_left_y, PATCH_SIZE)

			# Get the loss of the model for each patched version
			result = model.evaluate(patched_image.reshape(-1, 32, 32, 3), y_test[num].reshape(-1, 10), verbose=0)

			# Get the loss_map of the plot by computing the difference in loss from the original version to the patched value
			loss_map[top_left_y:top_left_y + PATCH_SIZE, top_left_x:top_left_x + PATCH_SIZE] = loss[0] - result[0]

	# Get the predicted label
	y_prob = model.predict(img.reshape(-1, 32, 32, 3))
	y_pred = cifar_dict[np.argmax(y_prob)]

	# Get true label	
	y_true = cifar_dict[int(np.where(y_test[num] == 1)[0])]

	# Plot the original image along with the difference in loss as a heatmap
	fig, ax = plt.subplots(1, 2, figsize=(15, 15))

	plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
	ax[0].imshow(img)
	ax[1].imshow(img, cmap='gray')
	im = ax[1].imshow(loss_map, cmap='Reds', alpha=0.3)
	fig.colorbar(im, fraction=0.05)
	ax[0].set_title("True Label: " + y_true.upper(), fontsize=15)
	ax[1].set_title("Predicted label with patch size " + str(PATCH_SIZE) + ": " + y_pred.upper(), fontsize=15)
