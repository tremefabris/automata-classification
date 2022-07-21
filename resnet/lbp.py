from utils.data	import *									# Já tem numpy as np
from skimage.feature import local_binary_pattern as LBP

# TODO:
#		Histograma do LBP tá meio estranho; talvez o número de bins esteja errado
#			>> PLOTTAR E ANALISAR O LBP E O HISTOGRAMA

def preprocess(data, target, info):
	img   = data[:, :, 0]
	img   = img * np.sqrt(info['test_var']) # Tentando desfazer a normalização de DataGenerator
	img   = img + info['test_mean']			# Tentando desfazer a normalização de DataGenerator
	#img   = img * 255.
	label = np.argmax(target)

	return img, label

def plot_hist(histogram, label, bins, range, where='data_transfer/'):
	plt.hist(histogram, bins=bins, range=range)
	plt.savefig(f'{where}HIST{label}.png', bbox_inches='tight')
	plt.close()

if __name__ == '__main__':

	data_info = get_datainfo()
	params = {'batch_size':128, 'dim':(120, 120), 'n_channels':1, 'n_classes':256, 'shuffle':True}
	partition, labels, folders = load_IDs_rules(validation_folder='../data/npz/cluster/')	# Validation servindo para pegar os 256 ECAs

	testing_generator = DataGenerator(partition['test'], labels, folders['test'], data_info['test_mean'], data_info['test_var'], **params)
	
	batch = testing_generator[0]
	for data, target in zip(batch[0], batch[1]):
		print(data.shape)
		print(target.shape)

		autom, label = preprocess(data, target, data_info)

		print(autom)
		print(label)

		lbp  = LBP(autom, 8, 1, method='uniform')
		hist = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10)) # TODO: tá fazendo sentido não kkkk
	
		plot_automata(autom, f'lbp/ORIGINAL{label}')
		plot_automata(lbp, f'lbp/LBP{label}')
		plot_hist(hist, label, np.arange(0, 11), (0, 10), where='data_transfer/lbp/')

	#print(hist)

	#print("\nFIM DA EXECUÇÃO CARALHO")
	print("\nFIM DA EXECUÇÃO")

