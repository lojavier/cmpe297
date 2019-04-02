#!python

from google_images_download import google_images_download
from imutils import paths
import argparse
import requests
import cv2
import os

response = google_images_download.googleimagesdownload()
arguments = {
	"prefix_keywords":None,
	"keywords":"car",
	"suffix_keywords":"coupe,sedan,minivan,suv",
	"format":"png",
	"limit":20,
	"print_urls":True
}
absolute_image_paths = response.download(arguments)
print(absolute_image_paths)

exit()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to output serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
print(args["detection_method"])
exit()

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []