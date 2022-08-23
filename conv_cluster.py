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


def build_model(input_shape):	# AlexNet-inspired model for CA classification
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
	model_fc.add(Dense(units=256, activation='softmax'))	# Segundo a recomendação do Scabini e usando apenas um layer FC
	
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
	return parser.parse_args()


if __name__ == '__main__':

	args = config_argparser()

	model, conv_pipe = build_model((120, 120, 1))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	print(f'LAYER 0 OF FULL MODEL: {model.get_layer(index=0).name}')

	conv_pipe.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	conv_pipe.summary()

	input("SAIR POR FAVOR")

	training_generator = NewDataGenerator(dataset_size=102400, n_channels=1)
	val_generator      = NewDataGenerator(dataset_size=20480, n_channels=1)
	testing_generator  = NewDataGenerator(dataset_size=20480, n_channels=1)

	cScaler_train = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=1)
	cScaler_val   = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=1)
	cScaler_test  = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=1)

	for idx, (images, _) in enumerate(training_generator):
		print(f"batch {idx} being processed")
		cScaler_train.continuous_train(images)
	training_generator.on_epoch_end()	# refills datagen
	training_generator.add_trained_scaler(cScaler_train)

	for idx, (images, _) in enumerate(val_generator):
		print(f"VAL: batch {idx} being processed")
		cScaler_val.continuous_train(images)
	val_generator.on_epoch_end()		# refills datagen
	val_generator.add_trained_scaler(cScaler_val)

	for idx, (images, _) in enumerate(testing_generator):
		print(f"TEST: batch {idx} being processed")
		cScaler_test.continuous_train(images)
	testing_generator.on_epoch_end()	# refills datagen
	testing_generator.add_trained_scaler(cScaler_test)

	if args.train is not None:
		stats = model.fit(x=training_generator, epochs=args.train, validation_data=val_generator,
	  					  use_multiprocessing=True, workers=6)
		model.evaluate(x=testing_generator, use_multiprocessing=True, workers=6)

		print("\n\nSAVING MODEL\n\n")
		model.save('models/alexnet')

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

	if args.test:
		model_test = keras.models.load_model('models/alexnet')

		minibatch   = testing_generator[0]
		test_images = minibatch[0][:10]
		test_labels = np.argmax(minibatch[1][:10], axis=1)
		print(test_images.shape)
		print(test_labels) 

		results = model_test(test_images, training=False)
		test_pred = np.argmax(results, axis=1)
		print(test_pred)

		for idx, img in enumerate(test_images):
			plot_automata(img, f'real{test_labels[idx]}_pred{test_pred[idx]}')


	if args.kmeans is not None:
		# TODO: Rodar para todo o test set (pegar snippet do resnet.py)

		params = {'batch_size':128, 'dim':(120, 120), 'n_channels':1, 'n_classes':256, 'shuffle':False}
		partition, labels, folders = load_IDs_rules()

		conv_generator = DataGenerator(partition['test'], labels, folders['test'], data_info['test_mean'], data_info['test_var'], **params)

		data_conv   = conv_generator[0][0]
		labels_conv = []
		for batch in conv_generator:
			labels_conv += np.argmax(batch[1], axis=1).tolist()
		print(data_conv.shape)

		#data  = np.array([])
		#rules = np.array([])
		#for i, (imgs, labels) in enumerate(conv_generator):	# Usando test set por causa do menor tamanho
		#	print(f'batch {i} loaded')
		#	if i == 0:
		#		data  = imgs
		#		rules = labels
		#	else:
		#		data  = np.concatenate((data, imgs), axis=0)
		#		rules = np.concatenate((rules, labels), axis=0)
		#	print(data.shape)

		print(f'{data.shape=}')
		print(f'{rules.shape=}')

		conv_dr = conv_pipe.predict(x=conv_generator, use_multiprocessing=True, workers=6)
		print(conv_dr.shape)

		pca = PCA(45, svd_solver='full')
		data_pca = pca.fit_transform(conv_dr)
		print(pca.explained_variance_ratio_)
		print("shape:", pca.explained_variance_ratio_.shape)

		ratio = pca.explained_variance_ratio_
		print("\n::::::::STATISTICS::::::::\n")
		print("sum of components:", np.sum(ratio))

		print(f'{np.argmax(pca.explained_variance_ratio_)}, {pca.explained_variance_ratio_[np.argmax(pca.explained_variance_ratio_)]}')
		print('\n::::::::::::::::::::::::\n')

		print(data_pca)
		print(data_pca.shape)

		print('\n::::::::::::::::::::::::\n')
		print("APROVEITANDO APENAS OS PRIMEIROS 45 COMPONENTES PARA CLUSTERING\n\n")


		labels = []
		inertias = []
		for k in range(2, args.kmeans):
			print(f"...initiating kmeans for k = {k}...")	
			km = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)#.fit(x)
			km_fit = km.fit_transform(data_pca)
			labels.append(km.labels_)
			inertias.append(km.inertia_)


		print("labels for k=4:", labels[2])
		print("shape:", labels[2].shape)
		print("real labels shape:", len(labels_conv))

		print()
		print("first 40 kmeans labels:", labels[2][:40])
		print("first 40 rules:        ", labels_conv[:40])

		print("inertias:", inertias)
		print("shape:", len(inertias))

		print('\n::::::::::::::::::::::::\n')

		# MAKING A SELECTION SYSTEM FOR THE CLUSTER-CLASSES

		view_number = 100		# Hardcoded just cause
		k_views = input("QUE k ANALISAR? ").split()
		k_views = [ int(x) for x in k_views ]

		for k in k_views:
			view_dict = {}

			for i in range(view_number):
				if labels[k-2][i] not in view_dict:
					view_dict[labels[k-2][i]] = []
				plot_automata(data_conv[i], f'{k}/rule{labels_conv[i]}_class{labels[k-2][i]}')
				view_dict[labels[k-2][i]].append(labels_conv[i])

			print(f"K = {k}\n")

			for n in view_dict:
				print(f"CLASSE {n}")

				for x in view_dict[n]:
					print(x, end=' ')
				print('\n')
			print()


	if args.view:

		params = {'batch_size':128, 'dim':(120, 120), 'n_channels':1, 'n_classes':256, 'shuffle':False}
		partition, labels, folders = load_IDs_rules()

		testing_generator = DataGenerator(partition['test'], labels, folders['test'], data_info['test_mean'], data_info['test_var'], **params)


		model_pred = keras.models.load_model('models/alexnet')
		labels_pred = model_pred.predict(x=testing_generator, use_multiprocessing=True, workers=6)

		# Transforming from one-hot to normal list
		labels_real = []
		for i, batch in enumerate(testing_generator):
			labels_real += np.argmax(batch[1], axis=1).tolist()
		labels_real = np.array(labels_real)

		labels_pred = np.argmax(labels_pred, axis=1)
		#print(labels_pred.shape)


		mislabels = [ i for i, x in enumerate(np.equal(labels_real, labels_pred)) if not x ]
		for i, index in enumerate(mislabels):
			print(f"MISLABEL No {i}")
			print(f"REAL VALUE: {labels_real[index]}")
			print(f"PRED VALUE: {labels_pred[index]}")
			print()
		print(f"TOTAL MISLABELS: {len(mislabels)}")

