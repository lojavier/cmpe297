#!python
# USAGE: python train_model.py --dataset vehicles --model vehicles.model --label-bin vehicles.pickle --plot vehicles_plot.png --batchsize 32 --epochs 25

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
# from pyimagesearch.vehiclenet import VehicleNet
# from pyimagesearch.smallvggnet import SmallVGGNet
# from pyimagesearch.resnet import ResNet
from pyimagesearch.stridednet import StridedNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.regularizers import l2
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", default="dataset_plot.png",
	help="path to output accuracy/loss plot")
ap.add_argument("-e", "--epochs", type=int, default=1,
	help="number of training epochs")
ap.add_argument("-b", "--batchsize", type=int, default=32,
	help="batch size")
args = vars(ap.parse_args())

# initialize our initial learning rate, # of epochs to train for,
# and batch size
# INIT_LR = 1e-2
INIT_LR = 1e-4
EPOCHS = int(args["epochs"])
BS = int(args["batchsize"])
IMG_SIZE = 64

# grab the image paths and randomly shuffle them
# initialize the data and labels
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
# random.seed(42)
random.shuffle(imagePaths)
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels (the required input
	# spatial dimensions of VehicleNet), and store the image in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
	data.append(image)

	# extract the class label from the image path and update the labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the optimizer and CNN model
print("[INFO] compiling model...")
# opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model = VehicleNet.build(width=IMG_SIZE, height=IMG_SIZE, depth=3, classes=len(lb.classes_))
model = StridedNet.build(width=IMG_SIZE, height=IMG_SIZE, depth=3, classes=len(lb.classes_), reg=l2(0.0005))
# model = SmallVGGNet.build(width=IMG_SIZE, height=IMG_SIZE, depth=3, classes=len(lb.classes_))
# model = MiniVGGNet.build(width=IMG_SIZE, height=IMG_SIZE, depth=3, classes=len(lb.classes_))
# model = ResNet.build(IMG_SIZE, IMG_SIZE, 3, numClasses, (2, 2, 3), (32, IMG_SIZE, 128, 256), reg=0.0005)
# if len(lb.classes_) > 2:
# 	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# elif len(lb.classes_) == 2:
# 	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# else:
# 	model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = EPOCHS
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

# save the model and label binarizer to disk
print("[INFO] serializing network to '{}' and label binarizer to '{}'...".format(args["model"],args["label_bin"]))
model.save(args["model"])
model.save_weights(args["dataset"]+"_weights.h5")
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()