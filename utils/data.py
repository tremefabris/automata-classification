import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tensorflow import keras
from glob import glob
import re

from elementar1D import run_automata as generate_automata


# TODO: Tirar o "New" quando terminar a implementação, colocar "Old" no DataGenerator antigo

# TODO:
#		Com essa implementação, eu tô gerando imagens novas a cada época. O diagrama da regra 110 de uma época
#		não é o mesmo do diagrama da próxima ou da anterior. Isso pode ser um problema (perguntar pro Scabini).
#		
#		Se isso for um problema, talvez eu possa salvar um array de shape (dataset_size // batch_size, dim[0])
#		com todos os impulsos iniciais e gerar os diagramas a cada época.
#
class NewDataGenerator(keras.utils.Sequence):	# I AM normalizing the data before outputting it

	def __init__(self, dataset_size, batch_size=256, dim=(120, 120), impulse='random', radius=1, nstates=2, n_channels=1, n_classes=256, shuffle=True):
		self.dataset_size  = dataset_size
		self.batch_size    = batch_size
		self.dim		   = dim
		self.impulse	   = impulse
		self.radius		   = radius
		self.nstates	   = nstates
		self.n_channels	   = n_channels
		self.n_classes	   = n_classes
		self.shuffle	   = shuffle
		self.dataset_shape = (dataset_size, *dim, n_channels)
		self.on_epoch_end()

	def __len__(self):
		return self.dataset_size // self.batch_size

	def __data_generation(self, batch_rules):

		X = cp.empty((self.batch_size, *self.dim, 1))
		y = cp.empty((self.batch_size), dtype=int)

		for i, rule in enumerate(batch_rules.get()):		# Converting to np.ndarray for compatibility with elementar1D
			tmp    = generate_automata(rule, *self.dim, self.impulse, self.radius, self.nstates)
			tmp    = cp.array(np.reshape(tmp.get(), (*self.dim, 1)))
			X[i, ] = tmp
			y[i]   = rule

		if self.n_channels != 1:
			X = cp.repeat(X, self.n_channels, -1)		# Se as imagens precisam ter outra quantidade de canais
		return X.get(), keras.utils.to_categorical(y.get(), num_classes=self.n_classes)

	def __getitem__(self, index):
		batch_round_rules = self.rules[index * self.batch_size : (index + 1) * self.batch_size]

		X, y = self.__data_generation(batch_round_rules)
		return X, y	

	def on_epoch_end(self):
		self.rules = cp.array([], dtype=int)
		for i in range(self.dataset_size // self.batch_size):
			self.rules = cp.concatenate((self.rules, cp.arange(self.batch_size)), axis=0)
		
		if self.shuffle == True:
			cp.random.shuffle(self.rules)

# TODO: Hora de refatorar DataGenerator
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
	plt.close(fig)

def save_image(img, *, directory='imgs/'):
	raise NotImplementedError

def plot_image(img_name, *, directory='imgs/', save=True):
	img = plt.imread(f'{directory}{img_name}.png')
	plt.imshow(img)
	plt.savefig(f'data_transfer/{img_name}.png') if save else plt.show()

if __name__ == '__main__':
	pass

