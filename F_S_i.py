import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
import os
import numpy as np



################################### FEATURE EXTRACTION (ImageNet pre-trained ResNet-50 till average pool layer) ###################################


def get_dir_contents_list(dir_path):
	"""
	:param dir_path: the path to the dir of images in a domain
	:return: a list of str names of each of the files contained within the dir located @ dir_path
	"""
	dir_contents_list = os.listdir(dir_path)
	return dir_contents_list

def load_domain_cls_images(domain_path, cls):
	"""
	:param domain_path: e.g. r"D:\Datasets\Office_31_data\Original_images\Original_images\amazon\images"
	:param cls: e.g. "back_pack"
	:return:
	"""
	domain_cls_path = os.path.join(domain_path, cls)
	domain_cls_img_paths = [os.path.join(domain_cls_path, img_file_name) for img_file_name in get_dir_contents_list(domain_cls_path)]
	num_domain_cls_images = len(domain_cls_img_paths)
	images_array = np.zeros((num_domain_cls_images, 224, 224, 3))
	for idx, img_path in enumerate(domain_cls_img_paths):
		img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		x_i = preprocess_input(img)
		images_array[idx] = x_i
	return images_array


# resnet <=> feature_extractor
# W/ include_top=False, we’re specifying that the last layer of the network we want to load is the bottleneck layer
# weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
#  input_shape: optional shape tuple, only to be specified if `include_top` is False (otherwise ResNet50 expects the input shape has to be `(224, 224, 3)`
input_shape = (224, 224, 3)
resnet_feat_extraction = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

print(resnet_feat_extraction.summary())


# Office31 has 4 domains and each domain has its own class of images.
domain_dir_path = r"D:\Datasets\Office_31_data\Original_images\Original_images\amazon\images"
cls = "back_pack"
domain_cls_preprocessed_images = load_domain_cls_images(domain_dir_path, cls)

features = resnet_feat_extraction.predict(domain_cls_preprocessed_images) # extracts the features

################################### FEATURE EXTRACTION (Lin FC layers using Keras Func API) ###################################

features_input =Input(shape=(2048, 1024))
lin_fc_1 = Dense(64, activation='elu')(features_input) # so here "features_input" is the input to the Dense layer (a lin fully connected layer w/ elu activation func)
lin_fc_2 = Dense(64, activation='elu')(lin_fc_1)
lin_fc_2_batch_normed = BatchNormalization()(lin_fc_2)
lin_fc_3 = Dense(64, activation='elu')(lin_fc_2_batch_normed)
lin_fc_4 = Dense(64, activation='elu')(lin_fc_3)
lin_fc_4_batch_normed = BatchNormalization()(lin_fc_4)

# build the model by supplying inputs/outputs
F_S_i = Model(inputs=features_input, outputs=lin_fc_4_batch_normed)
F_S_i.summary()
