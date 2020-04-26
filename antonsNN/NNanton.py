import sys, os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

from sklearn.datasets import make_classification, make_circles, make_moons, make_gaussian_quantiles
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.optim




class DummyDataGenerator():
	# TODO: normaliser?

	def __init__(self, n_samples=2e3):

		self.n_samples = int(n_samples)

	def split_data(self, feature_matrix, label_vec):

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=.5, random_state=420)

		for train_index, test_index in stratified_splitter.split(np.zeros(feature_matrix.shape[0]), label_vec):

			x_training, x_validation = feature_matrix[train_index], feature_matrix[test_index]
			y_training, y_validation = label_vec[train_index], label_vec[test_index]

		data_container = {
				'x_full': feature_matrix,
				'x_train': x_training,
				'x_test': x_validation,
				'y_full': label_vec,
				'y_train': y_training,
				'y_test': y_validation,
				}

		return data_container

	def get_standard_data(self, seperation=1., noise=.5, positives_ratio=.3):

			x, y = make_classification(
				n_samples=self.n_samples,
				n_features=2,
				n_informative=2,
				n_redundant=0,
				n_repeated=0,
				n_classes=2,
				n_clusters_per_class=2,
				weights=[positives_ratio, 1. - positives_ratio],
				flip_y=noise,
				class_sep=seperation,
				hypercube=True,
				shift=0.0,
				scale=1.0,
				shuffle=True,
				random_state=420,
				)

			return self.split_data(x, y)

	def get_circles_data(self, seperation=.8, noise=1.):

			x, y = make_circles(
				n_samples=int(self.n_samples),
				shuffle=True,
				noise=noise,
				random_state=420,
				factor=seperation,
				)

			return self.split_data(x, y)

	def get_moons_data(self, noise=.4):

			x, y = make_moons(
				n_samples=int(self.n_samples),
				shuffle=True,
				noise=noise,
				random_state=420,
				)

			return self.split_data(x, y)

	def get_gaussian_quantiles_data(self, covariance=.5):

			x, y = make_gaussian_quantiles(
				cov=covariance,
				n_samples=int(self.n_samples),
				n_features=2,
				n_classes=2,
				shuffle=True,
				random_state=420,
				)
			
			return self.split_data(x, y)

	def make_scatter_plot(self, dummy_data, figure_name):

		x0 = dummy_data['x_full'][dummy_data['y_full'] == 0]
		x1 = dummy_data['x_full'][dummy_data['y_full'] == 1]

		plt.figure(figsize=(12,12))

		plt.scatter(x0[:,0], x0[:,1], c='r', label='y=0')
		plt.scatter(x1[:,0], x1[:,1], c='g', label='y=1')

		plt.grid()
		plt.legend()
		plt.title(figure_name)
		plt.savefig(figure_name + '.pdf')
		plt.close()


class DataSetIterator(Dataset):

  def __init__(self, features, labels):

        self.labels = labels
        self.features = features

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):

        x = self.features[index, :]
        y = self.labels[index]

        return x, y


class AntonsNetwork(nn.Module):

	def __init__(self):

		super(AntonsNetwork, self).__init__()

		self.num_hidden_layers = 3
		self.layer_width = 64

		self.fully_connected_0 = nn.Linear(2, self.layer_width)
		self.fully_connected_1 = nn.Linear(self.layer_width, self.layer_width)
		self.fully_connected_2 = nn.Linear(self.layer_width, self.layer_width)
		self.fully_connected_3 = nn.Linear(self.layer_width, self.layer_width)
		self.fully_connected_final = nn.Linear(self.layer_width, 2)

		self.relu = F.relu
		self.sigmoid = sigmoid
		self.softmax = F.softmax


	def forward(self, x):
		# TODO dropout, other activations

		x = self.relu(self.fully_connected_0(x.float()))

		if self.num_hidden_layers > 1:
			x = self.relu(self.fully_connected_1(x))
			if self.num_hidden_layers > 2:
				x = self.relu(self.fully_connected_2(x))
				if self.num_hidden_layers > 3:
					x = self.relu(self.fully_connected_3(x))

		x = self.softmax(self.fully_connected_final(x), dim=1)

		return x


class NetworkTrainer():

	def __init__(self, epochs=1001, learning_rate=1e-2, batch_size=128, validation_freq=100, data_mode='standard'):

		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.validation_freq = validation_freq
		self.data_mode = data_mode

		self.train_loss_vec = []
		self.val_loss_vec = []
		self.epoch_counter_train = 0
		self.epoch_counter_val = []


		try: os.mkdir(data_mode)
		except FileExistsError: pass


		self.model = AntonsNetwork()
		self.criterion = nn.CrossEntropyLoss()

		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr=self.learning_rate, 
			betas=(0.9, 0.999), 
			eps=1e-08, 
			weight_decay=0, 
			amsgrad=False)


		data_generator = DummyDataGenerator()
		if self.data_mode == 'standard':
			self.loaded_data = data_generator.get_standard_data()
		elif self.data_mode == 'circles':
			self.loaded_data = data_generator.get_circles_data()
		elif self.data_mode == 'moons':
			self.loaded_data = data_generator.get_moons_data()
		elif self.data_mode == 'gauss':
			self.loaded_data = data_generator.get_gaussian_quantiles_data()
		else:
			print('\n\nerror! unavailable data_mode specified: ' + data_mode)


		self.training_generator = self.get_data_generators(self.loaded_data['x_train'], self.loaded_data['y_train'])
		self.validation_generator = self.get_data_generators(self.loaded_data['x_test'], self.loaded_data['y_test'])


		self.train()


	def get_data_generators(self, features, labels):

		set_iterator0 = DataSetIterator(features, labels)
		set_iterator = DataLoader(
			set_iterator0, 
			batch_size=self.batch_size, 
			shuffle=True, 
			num_workers=0, 
			drop_last=False)

		return set_iterator


	def evaluate(self):

		def predict(x):

			x = torch.from_numpy(x).float()
			out = self.model(x)
			out = out[:,0]
			return out.detach().numpy()

		def plot_decision_boundary(pred_func, X, y):
			# Set min and max values and give it some padding
			x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
			y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
			h = 0.0025
			# Generate a grid of points with distance h between them
			xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
			# Predict the function value for the whole gid
			Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
			Z = Z.reshape(xx.shape)
			# Plot the contour and training examples
			plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
			plt.colorbar()
			plt.scatter(X[:, 0], X[:, 1], s=4, c=y, cmap=plt.cm.binary)

		plt.figure('epoch ' + str(self.epoch_counter_train), figsize=(16, 8))

		plt.subplot(121)
		plot_decision_boundary(lambda x : predict(x), self.loaded_data['x_train'], self.loaded_data['y_train'])
		plt.title('training')
		
		plt.subplot(122)
		plot_decision_boundary(lambda x : predict(x), self.loaded_data['x_test'], self.loaded_data['y_test'])
		plt.title('validation')

		plt.savefig(self.data_mode + '/evaluation_epoch' + str(self.epoch_counter_train) + '.png')
		plt.close()



	def validate(self):

		validation_loss = 0.
		for local_batch, local_labels in self.training_generator:

			output = self.model(local_batch)
			# print(output.detach().numpy())
			loss = self.criterion(output, local_labels)
			validation_loss += loss / self.batch_size

		self.val_loss_vec.append(validation_loss.detach().numpy())
		self.epoch_counter_val.append(self.epoch_counter_train)

		print('validation score @epoch' + str(self.epoch_counter_train).ljust(25, '.') + str(np.round(validation_loss.detach().numpy(), 5)))


	def train(self):

		# TODO built early stopping criterion

		print('\n\n\nstarting training for ' + str(self.epochs) + ' epochs on ' + self.data_mode + ' dataset...\n')

		for _ in range(self.epochs):

			epoch_loss = 0.

			for local_batch, local_labels in self.training_generator:

				self.optimizer.zero_grad()

				output = self.model(local_batch)
				loss = self.criterion(output, local_labels)

				epoch_loss += loss / self.batch_size

				loss.backward()
				self.optimizer.step()

			if self.epoch_counter_train % self.validation_freq == 0:
				self.validate()
				self.evaluate()

			self.train_loss_vec.append(epoch_loss.detach().numpy())
			self.epoch_counter_train += 1


		print('\n\nfinished training\n\n')

		# gif_fps = 1
		# image_path = Path(self.data_mode)
		# images = list(image_path.glob('*.png'))
		# image_list = []
		# for file_name in images:
		# 	image_list.append(imageio.imread(file_name))
		# imageio.mimwrite(self.data_mode + '/training_progess.gif', image_list, duration=1/gif_fps)

		train_loss_plot = np.asarray(self.train_loss_vec)
		train_loss_axis = np.arange(self.epoch_counter_train)

		val_loss_plot = np.asarray(self.val_loss_vec)
		val_loss_axis = np.asarray(self.epoch_counter_val)

		plt.figure(figsize=(16,16))

		plt.subplot(121)
		plt.plot(train_loss_axis, train_loss_plot, c='r', label='training loss')
		# plt.ylim([.9 * np.amin(val_loss_plot), .2 * np.amax(train_loss_plot)])
		plt.yscale('log')
		plt.xlabel('epochs')
		plt.ylabel('crossentropy loss')
		plt.title('training')
		plt.grid()
		plt.legend()
		
		plt.subplot(122)
		plt.plot(val_loss_axis, val_loss_plot, c='g', label='training loss')
		# plt.ylim([.9 * np.amin(val_loss_plot), .2 * np.amax(train_loss_plot)])
		plt.yscale('log')
		plt.xlabel('epochs')
		plt.ylabel('crossentropy loss')
		plt.title('validation')
		plt.grid()
		plt.legend()

		plt.savefig(self.data_mode + '/training.pdf')
		plt.close()




NetworkTrainer(data_mode = 'standard')
NetworkTrainer(data_mode = 'circles')
NetworkTrainer(data_mode = 'moons')
NetworkTrainer(data_mode = 'gauss')




