from utils.data import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse

from tensorflow.keras.datasets import fashion_mnist

def build_simple_ae(input_shape):											# Construindo primeiro pro FASHION MNIST
	img		= keras.layers.Input(shape=input_shape)
	#e 		= keras.layers.Dense(units=2000, activation='relu')(img)		# Assume que o input já é flattened
	e 		= keras.layers.Flatten()(img)
	e 		= keras.layers.Dense(units=392, activation='relu')(e)
	encoded = keras.layers.Dense(units=78, activation='relu')(e)

	d       = keras.layers.Dense(units=392, activation='relu')(encoded)
	d		= keras.layers.Dense(units=784, activation='sigmoid')(d)
	decoded = keras.layers.Reshape((28, 28))(d)

	encoder	    = keras.Model(img, encoded)
	autoencoder = keras.Model(img, decoded)
	return autoencoder, encoder

def config_argparser():
	parser = argparse.ArgumentParser(description="SIMPLE AUTOENCODER FOR LEARNING")
	parser.add_argument('--batch', required=True, type=int, metavar='N', default=32, help="Tamanho do batch para treinamento")
	parser.add_argument('--test', type=int, metavar='N', help="Quantas vezes testar o autoencoder exibindo dados originais e reconstruídos")
	return parser.parse_args()

def load_preprocess():
	(x_train, _), (x_test, _) = fashion_mnist.load_data()

	x_train = x_train.astype('float32') / 255.
	x_test  = x_test.astype('float32') / 255.

	return x_train, x_test


if __name__ == '__main__':
	args = config_argparser()
	x_train, x_test = load_preprocess()

	autoencoder, encoder = build_simple_ae((28, 28))
	autoencoder.compile(optimizer='adam', loss='mse')
	autoencoder.fit(x_train, x_train, batch_size=args.batch, epochs=10, validation_data=(x_test, x_test))

	if args.test is not None:
		test_data = x_test[:args.test]
		embedded  = encoder.predict(test_data)
		reconst   = autoencoder.predict(test_data)

		for i, (o, r) in enumerate(zip(test_data, reconst)):
			print(embedded[i])
			print()

			plt.imshow(o)
			plt.savefig(f'data_transfer/ORIGINAL{i}_ae.png')
			plt.close()
			plt.imshow(r)
			plt.savefig(f'data_transfer/RECONSTRUCTED{i}_ae.png')
			plt.close()

