# USAGE
# python test_model.py --images vehicles --model vehicles.model --label-bin vehicles.pickle
# python test_model.py --images vehicles --model test/vehicles_25epoch/vehicles.model --label-bin test/vehicles_25epoch/vehicles.pickle --output test/vehicles_25epoch/test_model_results_25.jpg
# python test_model.py --images vehicles --model test/vehicles_50epoch/vehicles.model --label-bin test/vehicles_50epoch/vehicles.pickle --output test/vehicles_50epoch/test_model_results_50.jpg
# python test_model.py --images vehicles --model test/vehicles_75epoch/vehicles.model --label-bin test/vehicles_75epoch/vehicles.pickle --output test/vehicles_75epoch/test_model_results_75.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to out input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-f", "--flatten", type=int, default=-1,
	help="whether or not we should flatten the image")
ap.add_argument("-o", "--output", type=str, default="test_model_results.jpg",
	help="path to output directory results")
args = vars(ap.parse_args())

single_image=False

GRID_LAYOUT=5 if not single_image else 1
TILE_SIZE=150 if not single_image else 300
IMG_SIZE = 64

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# grab all image paths in the input directory and randomly sample them
imagePaths = list(paths.list_images(args["images"]))
if not single_image:
	random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:(GRID_LAYOUT*GRID_LAYOUT)]

# initialize our list of results
results = []

# loop over our sampled image paths
for p in imagePaths:
	# load our original input image
	orig = cv2.imread(p)

	# pre-process our image by converting it from BGR to RGB channel
	# ordering (since our Keras mdoel was trained on RGB ordering),
	# resize it to 64x64 pixels, and then scale the pixel intensities
	# to the range [0, 1]
	image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
	image = image.astype("float") / 255.0

	# order channel dimensions (channels-first or channels-last)
	# depending on our Keras backend, then add a batch dimension to
	# the image
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# make predictions on the input image
	preds = model.predict(image)
	pred = preds.argmax(axis=1)[0]
	pred_perc = preds[0][pred] * 100

	# an index of zero is the 'parasitized' label while an index of
	# one is the 'uninfected' label
	# label = "car" if pred == 0 else "not car"
	label = lb.classes_[pred]
	color = (0, 0, 255) if pred_perc < 80 else (0, 255, 0)

	# resize our original input (so we can better visualize it) and
	# then draw the label on the image
	orig = cv2.resize(orig, (TILE_SIZE, TILE_SIZE))
	text = "{}: {:.2f}%".format(label, pred_perc)
	cv2.putText(orig, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# add the output image to our list of results
	results.append(orig)

# create a montage using 128x128 "tiles" with 5 rows and 5 columns
montage = build_montages(results, (TILE_SIZE, TILE_SIZE), (GRID_LAYOUT, GRID_LAYOUT))[0]

# show the output montage
cv2.imwrite(args["output"], montage)
cv2.imshow("Results", montage)
cv2.waitKey(0)