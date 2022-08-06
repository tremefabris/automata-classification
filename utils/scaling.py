import numpy as np

class ContinuousScaler():
	def __init__(self, scaler_algorithm, image_shape, n_channels):
		self.scaler = scaler_algorithm
		#self.batch_size = batch_size
		self.image_shape = image_shape
		self.n_channels = n_channels

	def continuous_train(self, data):
		if len(data.shape) > 2:								# not (n samples, n features)
			data = np.reshape(data, (data.shape[0], -1))	# now (n samples, n features)
		self.scaler.partial_fit(data)

	# Talvez eu não precise dessa função; o próprio scaler já faz transform contínuo
	def continuous_transform(self, data):
		if len(data.shape) > 2:
			data = np.reshape(data, (data.shape[0], -1))
		transformed_data = self.scaler.transform(data)
		ready_data = np.reshape(transformed_data, (transformed_data.shape[0], *self.image_shape, self.n_channels))
		return ready_data


# Eu preciso alimentar o resnet com um keras.utils.Sequence. Então basicamente eu preciso de um fluxo da forma
# 		NewDataGen <--> ContinuousScaler --> resnet(NewDataGen)
# E se eu adicionar um método em NewDataGen chamado add_trained_scaler para que ele possa passar os diagramas
# criados pelo normalizador antes de cuspí-los
