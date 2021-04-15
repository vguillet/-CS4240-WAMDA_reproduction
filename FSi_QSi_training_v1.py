import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import models
from tensorflow.keras import layers

# from tensorflow.keras import o


# Domain #1 | S_1
source_domain = "amazon"
num_imgs = 2817

# Domain #2 | S_2
# source_domain = "dslr"
# num_imgs = 498

# # Domain #3 | S_3
# source_domain = "webcam"
# num_imgs = 795


val_split = 0.2
train_split = 0.8
bs = 32
epoks = 20
img_dataset_dir = f"Deep_learning_repoduction/Office_31_data/Original_images/Original_images/{source_domain}/images"
data_gen = ImageDataGenerator(rescale=1. / 255,
                              validation_split=val_split,
                              # rotation_range=20,
                              rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              # featurewise_center=True,
                              horizontal_flip=True,
                              fill_mode='nearest')

# img_dataset_dir = r"D:\Datasets\Office_31_data\Original_images\Original_images\amazon\images"
# img_dataset_dir = fr"D:\Datasets\Office_31_data\Original_images\Original_images\{source_domain}\images"
# data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# train_gen = data_gen.flow_from_directory(img_dataset_dir, target_size=(224, 224), subset="training", batch_size=bs, class_mode="categorical")
# val_gen = data_gen.flow_from_directory(img_dataset_dir, target_size=(224, 224), subset="validation", batch_size=bs, class_mode="categorical")

conv_base = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

# freezing layers
for resnet_50_base_layer_i in conv_base.layers:
    resnet_50_base_layer_i.trainable = False

conv_base.summary()


class PredictionCallback(Callback):
    def on_epoch_end(self, step, images_cnt):
        print('prediction: {0} at epoch: {1}'.format(step, images_cnt))


def extract_features(img_dataset_dir, sample_count, bs, subset, classes):
    """
    :param img_dataset_dir:
    :param sample_count: int(train_split*num_imgs) if subset="training" else int(val_split*num_imgs) subset="validation"
    :param bs: batch size
    :param subset: "training" or "validation".
    :param classes: list type of all class labels of the source domain S_i
    :return: ResNet50 features extracted from the source domain S_i
    """
    features = np.zeros(shape=(sample_count, 7, 7, 2048))  # latent vars
    labels = np.zeros(shape=(sample_count, len(classes)))  # latent vars
    subset_gen = data_gen.flow_from_directory(img_dataset_dir, target_size=(224, 224),
                                              subset=subset, batch_size=bs, class_mode="categorical",
                                              classes=classes, shuffle=False)

    print("subset_gen", subset_gen.input)

    # type(subset_gen) -> DirectoryIteration cls instance, Yields batches of images as (x, y) tuples
    i = 0  # iteration i
    step = 0
    images_cnt = 0  # tot cnt of images iterated feature extraction (.predict) over
    for inputs_batch, labels_batch in subset_gen:
        step = step + 1
        images_cnt = images_cnt + bs
        features_batch = conv_base.predict(inputs_batch, verbose=1, callbacks=[PredictionCallback()])
        # Sometimes encountering errors in the next 2 lines...
        # e.g. ould not broadcast input array from shape (89,7,7,2048) into shape (99,7,7,2048)
        features[i * bs: (i + 1) * bs] = features_batch
        labels[i * bs: (i + 1) * bs] = labels_batch
        i += 1
        if i * bs >= int(sample_count / bs):
            # break out before out of range thingy error
            break
    return features, labels


def get_dir_contents_list(dir_path):
    """
    :param dir_path: the path to the dir of images in a domain
    :return: a list of str names of each of the files contained within the dir located @ dir_path
    """
    dir_contents_list = os.listdir(dir_path)
    return dir_contents_list


classes = get_dir_contents_list(img_dataset_dir)

print("#######################      EXTRACTING TRAINING FEATURES     ##############################")

train_features, train_labels = extract_features(img_dataset_dir, int(num_imgs * train_split), bs, subset="training",
                                                classes=classes)

print("#######################      EXTRACTING VALIDATION FEATURES     ##############################")

validation_features, validation_labels = extract_features(img_dataset_dir, int(num_imgs * val_split), bs,
                                                          subset="validation", classes=classes)

print("#######################      RESHAPING EXTRACTED FEATURES     ##############################")

train_features = np.reshape(train_features, (int(num_imgs * train_split), 7 * 7 * 2048))
validation_features = np.reshape(validation_features, (int(num_imgs * val_split), 7 * 7 * 2048))

print("#######################      MODEL BUILDING     ##############################")

model = models.Sequential()
################### BASE MODEL ###################
# model.add(layers.Dense(128, activation='relu', input_dim=7 * 7 * 2048))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(31, activation='softmax'))
################### ######### ###################

###################### Q_S_i ######################
# model.add(Flatten()) <- APPARENTLY don't need

#  0.0001 = 1e-4
from tensorflow.keras import optimizers

adam_opt_lr = 0.0001
adam_opt = optimizers.Adam(learning_rate=adam_opt_lr)
# adam_opt = optimizers.Adam(lr=0.0001)
# optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.add(layers.Dense(2048, activation='elu', name="F_S_i-Layer_1", input_dim=7 * 7 * 2048))
model.add(layers.Dense(1024, activation='elu', name="F_S_i-Layer_2"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1024, activation='elu', name="F_S_i-Layer_3"))
model.add(layers.Dense(256, activation='elu', name="F_S_i-Layer_4"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(31, activation='softmax', name="Q_S_i-Layer_1"))
################### ######### ###################

model.compile(optimizer=adam_opt,
              loss="categorical_crossentropy",
              metrics=['accuracy'])

print("#######################      MODEL FITTING     ##############################")

history = model.fit(train_features, train_labels, epochs=epoks, batch_size=bs,
                    validation_data=(validation_features, validation_labels))

model.save(
    f"FSiQSi_{source_domain}_domain_Resnet50_and_basic_fc_dense_layer_trained_model_with_bs_of_{bs}_and_epochs_of_{epoks}")
