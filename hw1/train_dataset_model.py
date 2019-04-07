#!python

import matplotlib
matplotlib.use("Agg")

# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from pyimagesearch.resnet import ResNet
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path dataset of input images")
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

NUM_EPOCHS = 1
BS = 32

TRAIN_PATH = os.path.sep.join([args["dataset"], "train"])
TEST_PATH = os.path.sep.join([args["dataset"], "test"])
VAL_PATH = os.path.sep.join([args["dataset"], "validate"])

totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))

trainAug = ImageDataGenerator(	
	rescale = 1 / 255.0,
	rotation_range = 20,
	zoom_range = 0.05,
	width_shift_range = 0.05,
	height_shift_range = 0.05,
	shear_range = 0.05,
	horizontal_flip = True,
	fill_mode = 'nearest')

valAug = ImageDataGenerator(rescale = 1 / 255.0)

trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode = "categorical",
	target_size = (64, 64),
	color_mode = "rgb",
	shuffle = True,
	batch_size = BS)

valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode = "categorical",
	target_size = (64, 64),
	color_mode = "rgb",
	shuffle = True,
	batch_size = BS)

testGen = valGen.flow_from_directory(
	TEST_PATH,
	class_mode = "categorical",
	target_size = (64, 64),
	color_mode = "rgb",
	shuffle = True,
	batch_size = BS)

model = ResNet.build(64, 64, 3, 2, (2, 2, 3), (32, 64, 128, 256), reg=0.0005)
opt = SGD(lr=1e-1, momentum=0.9, decay=1e-1 / NUM_EPOCHS)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
# model.add(Conv2D(32, (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
# model.add(Flatten())
# model.add(Dense(units = 128, activation = 'relu'))
# model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# More than 2 classes
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

H = model.fit_generator(
	trainAug,
	steps_per_epoch = totalTrain // BS,
	validation_data = valGen,
	validation_steps = totalVal // BS,
	epochs = NUM_EPOCHS)

print("[INFO] dataset training successful!")

print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(totalTest // BS) + 1)

print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])
model.save_weights("model.h5")

# model_json = model.to_json()
# with open("./model.json","w") as json_file:
#   json_file.write(model_json)

N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])