import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from utils.data    import *
from utils.scaling import *

from sys import exit


def local_plot_automata(autom, name):
	plt.rcParams['image.cmap'] = 'binary'

	fig, ax = plt.subplots(figsize=(16, 9))
	ax.matshow(autom)
	ax.axis(False)
	plt.savefig(f'{name}.png', bbox_inches='tight')
	plt.close(fig)

def autoencoder(input_shape, embedding_size):
	img		= keras.layers.Input(shape=input_shape)
	e 		= keras.layers.Flatten()(img)							# Para a Resnet, serão 2048 pixels de saída
	e		= keras.layers.BatchNormalization()(e)
	e 		= keras.layers.Dense(units=2048, activation='relu')(e)
	e		= keras.layers.Dropout(0.25)(e)
	e		= keras.layers.BatchNormalization()(e)
	e		= keras.layers.Dense(units=1000, activation='relu')(e)
	e		= keras.layers.Dropout(0.25)(e)
	e 		= keras.layers.Dense(units=500, activation='relu')(e)
	encoded = keras.layers.Dense(units=embedding_size, activation='relu')(e)

	d 		= keras.layers.Dense(units=500, activation='relu')(encoded)
	d		= keras.layers.BatchNormalization()(d)
	d		= keras.layers.Dropout(0.25)(d)
	d       = keras.layers.Dense(units=1000, activation='relu')(d)
	d		= keras.layers.BatchNormalization()(d)
	d		= keras.layers.Dropout(0.25)(d)
	d		= keras.layers.Dense(units=2048, activation='relu')(d)
	decoded = keras.layers.Reshape(input_shape)(d)

	encoder	    = keras.Model(img, encoded)
	autoencoder = keras.Model(img, decoded)
	return autoencoder, encoder
	

def config_argparser():
	parser = argparse.ArgumentParser(description="ELEMENTARY CELLULAR AUTOMATA CLASSIFIER (Pretrained ResNet + KMeans)")

	parser.add_argument('--net', type=str, metavar='CNN', required=True, help="Especifica a rede neural a usar para extração de características")
	parser.add_argument('--kmeans', type=int, metavar='N', help="Realiza o agrupamento com K-means de 2 a N k's")

	exc_group1 = parser.add_mutually_exclusive_group()
	exc_group1.add_argument('--pca', type=float, metavar='N', help="Executa a redução de dimensionalidade com PCA usando N% da variância original")
	exc_group1.add_argument('--ae', type=int, metavar='L', help="Executa a redução de dimensionalidade com AutoEncoder com L neurônios de codificação")

	exc_group2 = parser.add_mutually_exclusive_group()
	exc_group2.add_argument('--clusters', nargs='+', type=int, metavar='', help="Define os k's para considerar para análise")
	exc_group2.add_argument('--eca', nargs='+', type=int, metavar='', help="Usando o ECA set, define os k's para considerar para análise")

	return parser.parse_args()

def dump_rules(clusters, assignments, rules, filename):	# WRAPPER FOR _dump_rules_cluster
	fd = open(filename, 'wt', encoding='utf-8')

	for k in clusters:
		fd.write(f"K = {k}\n\n")
		_dump_rules_cluster(assignments[k - 2], rules, fd)
		fd.write('\n')

	fd.close()

def _dump_rules_cluster(k_assignments, rules, fd):	# Escreve para todas classes e regras de um único k

	d  = {'cluster':k_assignments, 'rule':rules}
	df = pd.DataFrame(data=d)
	classes = df['cluster'].unique()

	for c in np.sort(classes):
		fd.write(f"CLASS {c}\n")
		groups = df[ df['cluster'] == c ].to_numpy()

		for group in np.sort(groups[:, 1]):
			fd.write(f'{group} ')
		fd.write('\n\n')



if __name__ == '__main__':
	args = config_argparser()

	eca_generator     = NewDataGenerator(dataset_size=256, n_channels=3)
	diagram_generator = NewDataGenerator(dataset_size=512000, n_channels=3)
	test_ae_generator = NewDataGenerator(dataset_size=51200, n_channels=3)


	if args.net.lower() in 'densenet':
		model = keras.applications.DenseNet121(include_top=False, input_shape=(120, 120, 3), pooling='avg')
	else:
		model = keras.applications.ResNet50(include_top=False, input_shape=(120, 120, 3), pooling='avg')

	if args.kmeans is not None:

		# Utilizando a nova classe ContinuousScaling
		cScaler = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=3)
		for idx, (images, _) in enumerate(diagram_generator):	# TODO: Para deploy, remover o enumerate
			print(f"batch {idx} is now processing")
			cScaler.continuous_train(images)

		diagram_generator.on_epoch_end()		# refills the datagenerator
		diagram_generator.add_trained_scaler(cScaler)
		
		extracted_features = model.predict(diagram_generator, use_multiprocessing=True, workers=4, verbose=1)

		print(f"{extracted_features.nbytes=}")
		print(f"{extracted_features.shape=}")
		print(f"{extracted_features[0]=}")
		print()

		if args.ae is not None:

			data_ae = MinMaxScaler().fit_transform(extracted_features)
			print(f"{data_ae.shape=}")

			autoenc, enc = autoencoder(data_ae.shape[1:], args.ae)

			autoenc.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
			autoenc.fit(x=data_ae, y=data_ae, batch_size=256, epochs=1000, validation_split=0.1) # Talvez colocar um milhão de autômatos pra treinar aqui

			# Avaliando os resultados do autoencoder
			test_ae_cScaler = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=3)
			
			for idx, (images, _) in enumerate(test_ae_generator):	# TODO: Para deploy, remover o enumerate
				print(f"TEST: batch {idx} is now processing")
				test_ae_cScaler.continuous_train(images)

			test_ae_generator.on_epoch_end()		# refills the datagenerator
			test_ae_generator.add_trained_scaler(test_ae_cScaler)
			print("Autoencoder evaluate")
			test_extfeat = model.predict(test_ae_generator, use_multiprocessing=True, workers=4, verbose=1)
			test_feedae  = MinMaxScaler().fit_transform(test_extfeat)
			autoenc.evaluate(x=test_feedae, y=test_feedae, batch_size=256, verbose=1)

			data_kmeans = enc.predict(data_ae, batch_size=256, verbose=1)

			print(f"{data_kmeans.shape=}")
			#exit()

		if args.pca is not None:
			pca 	    = PCA(0.9, svd_solver='full')
			data_kmeans = pca.fit_transform(extracted_features)		# Uai, agora tá dando 14 componentes kkkkkkk

			print(f"{pca.explained_variance_ratio_=}")
			print(f"{pca.explained_variance_ratio_.shape=}")
			print(f"{np.sum(pca.explained_variance_ratio_)=}")
			print(f"{data_kmeans=}")

		#input("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

		# K-means per se
		# Passei a armazenar todos os runs do K-means em um vetor pra poder usar depois pros ECAs
		kms = np.empty(args.kmeans, dtype=object)

		labels = []
		inertias = []
		for k in range(2, args.kmeans + 1):
			print(f"...initiating kmeans for {k} clusters...")
			kms[k - 2] = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42, verbose=0)
			km_fit = kms[k - 2].fit_transform(data_kmeans)		# Podia estar usando fit_predict()
			labels.append(kms[k - 2].labels_)
			inertias.append(kms[k - 2].inertia_)

		print(labels)
		print(inertias)

		#input("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

		if args.clusters is not None:	# Ou rodo pros clusters do testset, ou rodo pros 256 ECAs

			view_max = 512
			for k in args.clusters:
				for i in range(view_max):
					group = labels[k-2][i]		# k - 2 pois começo o k-means em k = 2
					rule  = rules[i]
					plot_automata(automs[i], f'cluster/{k}/rule{rule}_class{group}')

			dump_rules(args.clusters, labels, rules, 'data_transfer/assigned_labels_cluster.txt')
			print("\nIMAGES SAVED\n")

		# Fluxo para rodar os 256 Elementares
		if args.eca is not None:
			if args.pca is not None:
				save_dir = 'resnet/pca'		# plot_automata já inclui data_transfer como diretório inicial
			elif args.ae is not None:
				save_dir = 'resnet/ae'

			print()
			print("SENHORAS E SENHORES")
			print("AGORA É HORA")
			print("DOS EEEEEEECAAAAAAAAAAAAAAS")
			print()
			

			eca_cScaler = ContinuousScaler(StandardScaler(), image_shape=(120, 120), n_channels=3)
			for idx, (images, _) in enumerate(eca_generator):	# TODO: Para deploy, remover o enumerate
				print(f"ECA: batch {idx} is now processing")
				eca_cScaler.continuous_train(images)

			eca_generator.on_epoch_end()		# refills the datagenerator
			eca_generator.add_trained_scaler(cScaler)
		
			eca_features = model.predict(eca_generator, use_multiprocessing=True, workers=4, verbose=1)

			if args.ae is not None:
				data_ae = MinMaxScaler().fit_transform(eca_features)
				data_eca = enc.predict(data_ae, batch_size=256)

			if args.pca is not None:
				data_eca = pca.transform(eca_features)

			# TODO: Salvar as regras dos autômatos para plottar (pqp vei)

			# Salva as características extraídas pela ResNet dos 256 ECAs
			np.savez(f'data_transfer/{save_dir}/resnet_feat_extraction.npz', feats=eca_features, rules=rules)
			if args.eca is not None:
				np.savez(f'data_transfer/{save_dir}/ae_dr.npz', embedding=data_eca, rules=rules)

			eca_labels = []
			for k in range(2, args.kmeans):
				km_pred = kms[k - 2].predict(data_eca)
				eca_labels.append(km_pred)

			print(eca_labels)
			print(rules)

			for k in args.eca:
				print(f"PLOTTING FOR {k} CLUSTERS")
				for i in range(256):
					group = eca_labels[k-2][i]		# k - 2 pois começo o k-means em k = 2
					rule  = rules[i]
					plot_automata(gray_automs[i], f'{save_dir}/{k}/rule{rule}_class{group}')
			print("\nIMAGES SAVED\n")

			dump_rules(args.eca, eca_labels, rules, f'data_transfer/{save_dir}/assigned_labels.txt')

