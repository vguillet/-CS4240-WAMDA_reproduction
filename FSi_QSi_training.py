import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Deep_learning_repoduction.WAMDA_model_generator import WAMDA_model_generator
from tensorflow.keras.optimizers import Adam

# --> Set graphics card for Deep Learning
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
batch_size = 32
epochs = 100
learning_rate = 0.0001

img_dataset_dir = f"Deep_learning_repoduction/Office_31_data/Original_images/{source_domain}/images"

data_gen = ImageDataGenerator(rescale=1. / 255,
                              validation_split=val_split,
                              rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              # featurewise_center=True,
                              horizontal_flip=True,
                              fill_mode='nearest')

print("#######################      EXTRACTING TRAINING AND VALIDATION FEATURES     ##############################")
# --> Fetching training datatset
training_set_generator = data_gen.flow_from_directory(img_dataset_dir,
                                                      target_size=(224, 224),
                                                      subset="training",
                                                      batch_size=batch_size,
                                                      class_mode="categorical",
                                                      classes=os.listdir(img_dataset_dir),
                                                      shuffle=False)

# --> Fetching validation datatset
validation_set_generator = data_gen.flow_from_directory(img_dataset_dir,
                                                        target_size=(224, 224),
                                                        subset="validation",
                                                        batch_size=batch_size,
                                                        class_mode="categorical",
                                                        classes=os.listdir(img_dataset_dir),
                                                        shuffle=False)

print("#######################      MODEL BUILDING     ##############################")
fsi_qsi = WAMDA_model_generator(input_shape=(224, 224, 3),
                                optimiser=Adam(learning_rate=learning_rate),
                                run_mode="FQ",
                                resnet_trainable=False)

print("#######################      MODEL FITTING     ##############################")
history = fsi_qsi.model.fit(training_set_generator,
                            validation_data=validation_set_generator,
                            epochs=epochs,
                            steps_per_epoch=training_set_generator.n/training_set_generator.batch_size,
                            validation_steps=validation_set_generator.n/validation_set_generator.batch_size,
                            verbose=2)

fsi_qsi.save(domain=source_domain, bs=batch_size, epochs=epochs)
