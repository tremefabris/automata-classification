import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import re
import argparse

# TODO:
#		Consigo visualizar os agrupamentos que ele fez e preciso fazer algumas coisas a mais.
#		2. Avaliar mais bonitinho as imagens geradas
#		3. Entender o motivo dele estar classificando o 110 em duas classes diferentes sempre
#


class DataGenerator(keras.utils.Sequence):	# I AM normalizing the data before outputting it

	def __init__(self, list_IDs, labels, folder, mean, var, batch_size=32, dim=(120, 120), n_channels=1, n_classes=256, shuffle=True):
		self.list_IDs 	= list_IDs
		self.labels   	= labels
		self.folder		= folder
		self.batch_size = batch_size
		self.dim		= dim
		self.n_channels	= n_channels
		self.n_classes	= n_classes
		self.shuffle	= shuffle
		self.mean		= mean
		self.var		= var
		self.on_epoch_end()

	def __len__(self):
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __data_generation(self, list_IDs_temp):

		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size), dtype=int)

		for i, ID in enumerate(list_IDs_temp):
			with np.load(f'{self.folder}rule{ID}.npz') as npz:
				X[i, ] = np.reshape(npz['automata'], (*self.dim, self.n_channels))
			y[i] = self.labels[ID]

		X = np.repeat(self.normalize(X), 3, -1)		# Mudanças para o caso específico da ResNet; fingindo que as imgs são RGB
		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

	def __getitem__(self, index):
		indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		X, y = self.__data_generation(list_IDs_temp)
		return X, y	

	def normalize(self, X):
		return (X - self.mean) / np.sqrt(self.var)

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)


def load_IDs_rules(train_folder='../data/npz/', validation_folder='../data/npz/validation/', test_folder='../data/npz/test/'):

	folders = {'train': train_folder, 'validation': validation_folder, 'test': test_folder}

	train_filelist = glob(f"{folders['train']}*.npz")
	val_filelist   = glob(f"{folders['validation']}*.npz")
	test_filelist  = glob(f"{folders['test']}*.npz")
	
	list_IDs = {'train':[], 'validation':[], 'test':[]}
	rules    = {}

	for ftrain in train_filelist:
		ID_train = re.findall(r'\d+\_\d+', ftrain)[0]
		list_IDs['train'].append(ID_train)
		rule_train = int(ID_train.split('_')[0])
		rules[ID_train] = rule_train

	for fval in val_filelist:
		ID_val   = re.findall(r'\d+\_\d+', fval)[0]
		list_IDs['validation'].append(ID_val)
		rule_val   = int(ID_val.split('_')[0])
		rules[ID_val]   = rule_val

	for ftest in test_filelist:
		ID_test  = re.findall(r'\d+\_\d+', ftest)[0]
		list_IDs['test'].append(ID_test)
		rule_test  = int(ID_test.split('_')[0])
		rules[ID_test]  = rule_test

	print("\n")												# PARA TESTE
	print(f"TRAIN SIZE: {len(list_IDs['train'])}")			# PARA TESTE
	print(f"VALID SIZE: {len(list_IDs['validation'])}")		# PARA TESTE
	print(f"TEST SIZE: {len(list_IDs['test'])}")			# PARA TESTE

	return list_IDs, rules, folders

def get_datainfo():		# Hardcoded values for datasets' mean and variance
	train_mean = 0.49994439714655037
	train_var  = 0.24999999690833247
	val_mean   = 0.5001572943793412
	val_var    = 0.24999997525848155
	test_mean  = 0.4998299865722653
	test_var   = 0.2499999710954371
	return {'train_mean':train_mean, 'train_var':train_var, 'val_mean':val_mean, 'val_var':val_var, 'test_mean':test_mean, 'test_var':test_var}

def plot_automata(autom, name):
	plt.rcParams['image.cmap'] = 'binary'

	fig, ax = plt.subplots(figsize=(16, 9))
	ax.matshow(autom)
	ax.axis(False)
	plt.savefig(f'data_transfer/{name}.png', bbox_inches='tight')
	plt.close(fig)	# TODO: VER SE DÁ CERTO

def config_argparser():
	parser = argparse.ArgumentParser(description="ELEMENTARY CELLULAR AUTOMATA CLASSIFIER (Pretrained ResNet + KMeans)")

	# Seleção da CNN para feature extraction
	parser.add_argument('--net', type=str, metavar='CNN', required=True, help="Especifica a rede neural a usar para extração de características")

	# Para a visualização da feature extraction
	parser.add_argument('--view', type=int, metavar='N', help='Visualizar a extração de características de N imagens pela pretrained ResNet')
	parser.add_argument('--save', action='store_true', help='Salva as imagens geradas pelo fluxo de execução no dock data_transfer')

	# Para o agrupamento com K-means
	parser.add_argument('--kmeans', type=int, metavar='N', help="Realiza o agrupamento com K-means de 2 a N k's")

	# Testset clustering ou ECA clustering
	exc_group = parser.add_mutually_exclusive_group()
	exc_group.add_argument('--clusters', nargs='+', type=int, metavar='', help="Define os k's para considerar para análise")
	exc_group.add_argument('--eca', nargs='+', type=int, metavar='', help="Usando o ECA set, define os k's para considerar para análise")

	return parser.parse_args()

# TODO: Implementar dump de regras separando-as por classes
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

	# Data pipelines and necessary information for preprocessing
	data_info = get_datainfo()
	params = {'batch_size':128, 'dim':(120, 120), 'n_channels':1, 'n_classes':256, 'shuffle':True}
	partition, labels, folders = load_IDs_rules(validation_folder='../data/npz/cluster/')	# Validation servindo para pegar os 256 ECAs

	eca_generator 	  = DataGenerator(partition['validation'], labels, folders['validation'], data_info['val_mean'], data_info['val_var'], **params)
	testing_generator = DataGenerator(partition['test'], labels, folders['test'], data_info['test_mean'], data_info['test_var'], **params)

	# Avaliando empiricamente, constatei que o pooling='max' exalta mais alguns pixels das features. Vou mantê-lo por enquanto...
	if args.net.lower() in 'densenet':
		model = keras.applications.DenseNet121(include_top=False, input_shape=(120, 120, 3), pooling='max')
	else:
		model = keras.applications.ResNet50(include_top=False, input_shape=(120, 120, 3), pooling='max')


	if args.view is not None:
		data = training_generator[0]
		imgs = data[0][:args.view]
		labels = np.argmax(data[1][:args.view], axis=1)	# Transformando de one-hot pra inteiro

		results = model(imgs)
		results = np.reshape(results, (-1, 64, 32))		# Pra plottar nos .pngs

		if args.save:									# Meio redundante?
			for autom, label, feats in zip(imgs, labels, results):
				plot_automata(autom, f'rule{label}_img')
				plot_automata(feats, f'rule{label}_feat')
			print("\nIMAGES SAVED\n")

	if args.kmeans is not None:

		for i, (imgs, labels) in enumerate(testing_generator):	# Esse laço ficou bem ineficiente (memory- and performance-wise)
			print(f"batch {i} sendo processado")
			labels = np.argmax(labels, axis=1)
			if i == 0:
				automs  = imgs[:, :, :, 0]						# O slicing é pra eliminar os canais extras (RGB -> grayscale)

				results = model.predict_on_batch(imgs)
				rules   = labels
			else:
				automs  = np.concatenate((automs, imgs[:, :, :, 0]), axis=0)

				temp    = model.predict_on_batch(imgs)
				results = np.concatenate((results, temp), axis=0)
				rules   = np.concatenate((rules, labels), axis=0)

		print(f'{results.shape}')
		print(f'{automs.shape}')
		print(f'{rules.shape}')

		pca 	   = PCA(0.9, svd_solver='full')
		components = pca.fit_transform(results)		# Até agora sempre foram 55 componentes principais buscando 90%

		print(pca.explained_variance_ratio_)
		print(pca.explained_variance_ratio_.shape)
		print(np.sum(pca.explained_variance_ratio_))
		print(components)

		# K-means per se
		# Passei a armazenar todos os runs do K-means em um vetor pra poder usar depois pros ECAs
		kms = np.empty(args.kmeans, dtype=object)

		labels = []
		inertias = []
		for k in range(2, args.kmeans + 1):
			print(f"...initiating kmeans for {k} clusters...")
			kms[k - 2] = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42, verbose=0)
			km_fit = kms[k - 2].fit_transform(components)		# Podia estar usando fit_predict()
			labels.append(kms[k - 2].labels_)
			inertias.append(kms[k - 2].inertia_)

		print(labels)
		print(inertias)


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
			del automs, results, rules
			for i, (imgs, labels) in enumerate(eca_generator):
				print(f"ECA batch {i} sendo processado")
				labels = np.argmax(labels, axis=1)
				if i == 0:
					automs  = imgs[:, :, :, 0]						# O slicing é pra eliminar os canais extras (RGB -> grayscale)

					results = model.predict_on_batch(imgs)
					rules   = labels
				else:
					automs  = np.concatenate((automs, imgs[:, :, :, 0]), axis=0)

					temp    = model.predict_on_batch(imgs)
					results = np.concatenate((results, temp), axis=0)
					rules   = np.concatenate((rules, labels), axis=0)

			eca_components = pca.transform(results)

			eca_labels = []
			for k in range(2, args.kmeans):
				km_pred = kms[k - 2].predict(eca_components)
				eca_labels.append(km_pred)

			print(eca_labels)
			print(rules)

			view_max = 100
			for k in args.eca:
				print(f"PLOTTING FOR {k} CLUSTERS")
				for i in range(256):
					group = eca_labels[k-2][i]		# k - 2 pois começo o k-means em k = 2
					rule  = rules[i]
					plot_automata(automs[i], f'eca/{k}/rule{rule}_class{group}')
			print("\nIMAGES SAVED\n")

			dump_rules(args.eca, eca_labels, rules, 'data_transfer/assigned_labels.txt')



