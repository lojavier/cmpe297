#!python

# wget https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/face-recognition-opencv/face-recognition-opencv.zip
# wget https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/pi-face-recognition/pi-face-recognition.zip
# wget https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/deep-learning-google-images/google-images-deep-learning.zip

# Step 1: Import the required packages
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from imutils import paths
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path dataset of input images")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
args = vars(ap.parse_args())

NUM_EPOCHS = 2
BS = 32

TRAIN_PATH = os.path.sep.join([args["dataset"], "train"])
TEST_PATH = os.path.sep.join([args["dataset"], "test"])
totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

# Step 2: Initialising the CNN
model = Sequential()

# Step 3: Convolution
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 4: Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 5: Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 6: Flattening
model.add(Flatten())

# Step 7: Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Step 8: Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# model.fit(trainX, trainY, epochs = NUM_EPOCHS)

# Step 9: ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1 / 255.0,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1 / 255.0)

# Step 10: Load the training Set
training_set = train_datagen.flow_from_directory(TRAIN_PATH,
												target_size = (64, 64),
												batch_size = BS,
												class_mode = 'binary')

test_set = test_datagen.flow_from_directory(TEST_PATH,
											target_size = (64, 64),
											batch_size = BS,
											class_mode = 'binary')

# Step 11: Classifier Training 
H = model.fit_generator(training_set,
					steps_per_epoch = totalTrain // BS,
					epochs = NUM_EPOCHS,
					validation_data = test_set,
					validation_steps = totalTest // BS)

print("[INFO] dataset training successful!")

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
test_set.reset()
predIdxs = model.predict_generator(test_set, steps=(totalTest // BS) + 1)

# show a nicely formatted classification report
print(classification_report(test_set.classes, predIdxs, target_names=test_set.class_indices.keys()))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# Step 12: Convert the Model to json
model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

# Step 13: Save the weights in a seperate file
model.save_weights("model.h5")

# plot the training loss and accuracy
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
