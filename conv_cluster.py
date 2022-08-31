import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization, Flatten, Input, Dense

from sklearn.decomposition import PCA
from sklearn.cluster       import KMeans
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils.data    import *
from utils.scaling import *

from sys import exit


def build_model(input_shape):
	model_conv = keras.Sequential()
	model_conv.add(Input(input_shape))

	model_conv.add(Conv2D(filters=48, kernel_size=11, strides=4, padding='same'))
	model_conv.add(BatchNormalization())
	model_conv.add(Activation('relu'))
	model_conv.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

	model_conv.add(Conv2D(filters=96, kernel_size=5, strides=1, padding='same'))
	model_conv.add(BatchNormalization())
	model_conv.add(Activation('relu'))
	model_conv.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

	model_conv.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
	model_conv.add(BatchNormalization())
	model_conv.add(Activation('relu'))

	model_conv.add(Conv2D(filters=126, kernel_size=3, strides=1, padding='same'))
	model_conv.add(BatchNormalization())
	model_conv.add(Activation('relu'))

	model_conv.add(Conv2D(filters=48, kernel_size=3, strides=1, padding='same'))
	model_conv.add(BatchNormalization())
	model_conv.add(Activation('relu'))
	model_conv.add(MaxPooling2D(pool_size=4, strides=4, padding='same'))
	model_conv.add(Flatten())


	model_fc = keras.Sequential()
	model_fc.add(Dense(units=256, activation='softmax'))
	
	full_model = keras.Sequential([
		model_conv,
		model_fc
	])

	return full_model, model_conv


def plot_automata(autom, name):
	plt.rcParams['image.cmap'] = 'binary'

	fig, ax = plt.subplots(figsize=(16, 9))
	ax.matshow(autom)
	ax.axis(False)
	plt.savefig(f'data_transfer/{name}.png', bbox_inches='tight')

def config_argparser():
	parser = argparse.ArgumentParser(description="CLASSIFICADOR DE AUTÔMATOS CELULARES (ConvNet + KMeans)")
	parser.add_argument('--train', type=int, metavar='', help='Se deve-se treinar a rede convolucional novamente')
	parser.add_argument('--test', action='store_true', help='Se deve-se predictar as regras do conjunto de teste')
	parser.add_argument('--view', action='store_true', help='Se deve-se visualizar os erros na classificação')
	parser.add_argument('--kmeans', type=int, metavar='', help='Se deve-se agrupar o conjunto de treino convolucionado com K-means')
	parser.add_argument('--eca', nargs='+', type=int, metavar='', help='Quais k\'s dos ECAs imprimir')
	return parser.parse_args()


if __name__ == '__main__':

	args = config_argparser()

	model, conv_pipe = build_model((120, 120, 1))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	conv_pipe.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	conv_pipe.summary()


	training_generator = NewDataGenerator(dataset_size=102400, n_channels=1)
	val_generator      = NewDataGenerator(dataset_size=20480, n_channels=1)
	testing_generator  = NewDataGenerator(dataset_size=20480, n_channels=1)

	cScaler_train = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=1)
	cScaler_val   = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=1)
	cScaler_test  = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=1)

	for idx, (images, _) in enumerate(training_generator):
		cScaler_train.continuous_train(images)
	training_generator.on_epoch_end()	# refills datagen
	training_generator.add_trained_scaler(cScaler_train)

	for idx, (images, _) in enumerate(val_generator):
		cScaler_val.continuous_train(images)
	val_generator.on_epoch_end()		# refills datagen
	val_generator.add_trained_scaler(cScaler_val)

	for idx, (images, _) in enumerate(testing_generator):
		cScaler_test.continuous_train(images)
	testing_generator.on_epoch_end()	# refills datagen
	testing_generator.add_trained_scaler(cScaler_test)

	if args.train is not None:
		stats = model.fit(x=training_generator, epochs=args.train, validation_data=val_generator,
	  					  use_multiprocessing=True, workers=6)
		model.evaluate(x=testing_generator, use_multiprocessing=True, workers=6)

		print(f"Std deviation from train loss: {np.std(stats.history['loss'])}")
		print(f"Std deviation from train acc:  {np.std(stats.history['accuracy'])}")
		print(f"Std deviation from val loss:   {np.std(stats.history['val_loss'])}")
		print(f"Std deviation from val acc:    {np.std(stats.history['val_accuracy'])}")
		print()
		print()
		print(f"Train loss: {stats.history['loss']}")
		print(f"Train acc: {stats.history['accuracy']}")
		print(f"Val loss: {stats.history['val_loss']}")
		print(f"Val acc: {stats.history['val_accuracy']}")

	
	if args.kmeans is not None:

		eca_generator = NewDataGenerator(dataset_size=256, n_channels=1)
		cScaler_eca   = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=1)
		for idx, (images, _) in enumerate(eca_generator):
			print(f"ECA: batch {idx} being processed")
			cScaler_eca.continuous_train(images)
		eca_generator.on_epoch_end()	# refills datagen
		eca_generator.add_trained_scaler(cScaler_eca)
		
		data_eca_gen = eca_generator[0]
		imgs_conv_pipe  = data_eca_gen[0]
		rules_conv_pipe = data_eca_gen[1]
		eca_feats = conv_pipe.predict(imgs_conv_pipe)

		pca         = PCA(0.9, svd_solver='full')
		data_kmeans = pca.fit_transform(eca_feats)

		print(f"{pca.explained_variance_ratio_=}")
		print(f"{pca.explained_variance_ratio_.shape=}")
		print(f"{np.sum(pca.explained_variance_ratio_)=}")
		print(f"{data_kmeans=}")


		kms = np.empty(args.kmeans, dtype=object)

		labels = []
		inertias = []
		for k in range(2, args.kmeans + 1):
			kms[k - 2] = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42, verbose=0)
			km_fit = kms[k - 2].fit_transform(data_kmeans)
			labels.append(kms[k - 2].labels_)
			inertias.append(kms[k - 2].inertia_)

		print(inertias)

		if args.eca is not None:
			for k in args.eca:
				print(f"FOR {k} CLUSTERS")
				for i in range(256):
					rule = np.argmax(rules_conv_pipe[i])
					print(f'RULE: {rule}, CLASS: {labels[k - 2][i]}')
				print('\n\n')

