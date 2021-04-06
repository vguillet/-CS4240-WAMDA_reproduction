import tensorflow

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
import os
import sys
import numpy as np


class WAMDA_model_generator:
	def __init__(self, input_shape, optimiser, run_mode="FQ", resnet_trainable=False):
		"""

		:param input_shape: Tuple (shape????)
		:param run_mode: FQ, FD, FD_frozen, FEE
		"""
		self.run_mode = run_mode
		self.resnet_trainable = resnet_trainable

		# --> Record Model properties
		self.input_shape = input_shape

		# --> Build network
		self.model = None

		# ---- Pre-adaptation
		if run_mode == "FQ":
			print("Run mode: F_S_i -> Q_S_i")
			self.add_fsi()
			self.add_qsi()

			# self.model._name = "F_S_i-Q_S_i"

			self.model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])

		elif run_mode == "FD":
			print("Run mode: F_S_i (frozen) -> D_S_i")
			self.load_fsi()
			self.add_dsi()

			# self.model._name = "F_S_i_frozen-D_S_i"

			self.model.compile(loss="binary_crossentropy", optimizer=optimiser, metrics=["accuracy"])

		elif run_mode == "FD_frozen":
			print("Run mode: F_S_i (frozen) -> D_S_i (frozen)")
			self.load_fsi_dsi()

			# self.model._name = "F_S_i_frozen-D_S_i_frozen"

			self.model.compile(loss="binary_crossentropy", optimizer=optimiser, metrics=["accuracy"])

		# ---- WAMDA
		elif run_mode == "FEE":
			self.add_esi()
			self.add_et()

		else:
			sys.exit("Incorrect run mode")

		self.model.summary()

	# --------------------------------------------------------------------------------------------- Pre-adaptation
	def save(self, domain=None, bs=None, epochs=None):
		if bs is None and epochs is None:
			self.model.save(self.run_mode + "_model")

		else:
			self.model.save(domain + "_" + self.run_mode + f"_model_with_bs_of_{bs}_and_epochs_of_{epochs}")

		return

	# ----------------------------------------- FSI
	def add_fsi(self):
		# --> Create sequential model
		self.model = Sequential()

		# --> Add resnet50 to sequential model
		resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

		# Freeze resnet50 layers if need be
		if self.resnet_trainable is False:
			for resnet_50_base_layer_i in resnet_model.layers:
				resnet_50_base_layer_i.trainable = False

		self.model.add(resnet_model)
		self.model.add(Flatten())		# TODO: Evaluate flatten need with mentor

		# --> Add layer 1
		self.model.add(Dense(2048, activation='elu', name="F_S_i-Layer_1"))  # so here "features_input" is the input to the Dense layer (a lin fully connected layer w/ elu activation func)

		# --> Add layer 2
		self.model.add(Dense(1024, activation='elu', name="F_S_i-Layer_2"))
		self.model.add(BatchNormalization())

		# --> Add layer 3
		self.model.add(Dense(1024, activation='elu', name="F_S_i-Layer_3"))

		# --> Add layer 4
		self.model.add(Dense(256, activation='elu', name="F_S_i-Layer_4"))
		self.model.add(BatchNormalization())

		return

	def load_fsi(self):
		base_model = load_model('FQ_model')

		self.model = Sequential()

		for layer in base_model.layers[:-1]:
			layer.trainable = False
			self.model.add(layer)
		return

	# ----------------------------------------- QSI
	def add_qsi(self):
		self.model.add(Dense(31, activation='softmax', name="Q_S_i-Layer_1"))
		return

	# ----------------------------------------- DSI
	def add_dsi(self):
		self.model.add(Dense(256/2, activation='elu', name="D_S_i-Layer_1"))
		self.model.add(Dense(2, activation='sigmoid', name="D_S_i-Layer_2"))
		return

	def load_fsi_dsi(self):
		base_model = load_model('FD_model')

		self.model = Sequential()

		for layer in base_model.layers[:-1]:
			layer.trainable = False
			self.model.add(layer)

		return

	# --------------------------------------------------------------------------------------------- WAMDA
	def add_esi(self):
		self.model.add(Dense(1024, activation='elu'))
		self.model.add(BatchNormalization())

		self.model.add(Dense(1024, activation='elu'))
		self.model.add(BatchNormalization())

		self.model.add(Dense(1024, activation='elu'))
		self.model.add(BatchNormalization())

		self.model.add(Dense(256, activation='elu'))
		self.model.add(BatchNormalization())

		return

	def add_et(self):
		resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

		# --> Add resnet50 layers to model
		for layer in resnet50_model.layers:
			layer.trainable = False
			self.model.add(layer)

		# --> Add layer 1
		self.model.add(Dense(1024, activation='elu'))

		# --> Add layer 2
		self.model.add(Dense(1024, activation='elu'))
		self.model.add(BatchNormalization())

		# --> Add layer 3
		self.model.add(Dense(1024, activation='elu'))
		self.model.add(BatchNormalization())

		# --> Add layer 4
		self.model.add(Dense(256, activation='elu'))
		self.model.add(BatchNormalization())

		return


if __name__ == "__main__":
	# --> Step 1
	pre_adaptation_step_1 = WAMDA_model_generator(input_shape=(224, 224, 3), run_mode="FQ")

	pre_adaptation_step_1.save()

	# --> Step 2
	pre_adaptation_step_2 = WAMDA_model_generator(input_shape=(224, 224, 3), run_mode="FD")

	pre_adaptation_step_2.save()

	# --> Step 3
	pre_adaptation_step_3 = WAMDA_model_generator(input_shape=(224, 224, 3), run_mode="FD_frozen")

	pre_adaptation_step_3.save()
