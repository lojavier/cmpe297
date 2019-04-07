#!/bin/bash

# wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
# tar -xvzf car_ims.tgzr
# mv -v car_ims vehicle
# dataset=(`ls vehicle/*.jpg`)
# if [[ $? -ne 0 ]]; then
# 	exit 1
# fi
# dataset_total=${#dataset[@]}
# lower_limit=$((dataset_total / 3))
# # lower_limit=200
# upper_limit=$((lower_limit * 2))
# mkdir -vp 'vehicle/training/car'
# mkdir -vp 'vehicle/validation/car'
# mkdir -vp 'vehicle/testing/car'
# # for ((i=0; i<${#dataset[@]}; i++)); do
# for ((i=0; i<600; i++)); do
# 	if [[ $i -lt $lower_limit ]]; then
# 		mv -v ${dataset[$i]} 'vehicle/training/car'
# 	elif [[ $i -gt $upper_limit ]]; then
# 		mv -v ${dataset[$i]} 'vehicle/validation/car'
# 	else
# 		mv -v ${dataset[$i]} 'vehicle/testing/car'
# 	fi
# done
# exit 0

# wget https://www.kaggle.com/c/image-classification2/download/imagecl.tar.gz
# tar -xvzf imagecl.tar.gz
# mv -v imagecl vehicle

cp -vr imagecl vehicle
cp -vr 'vehicle/train' 'vehicle/training'
cp -vr 'vehicle/train' 'vehicle/validation'
cp -vr 'vehicle/train' 'vehicle/testing'


dataset_total=${#dataset[@]}
lower_limit=$((dataset_total / 3))

upper_limit=$((lower_limit * 2))
mkdir -vp 'vehicle/training/car'
mkdir -vp 'vehicle/validation/car'
mkdir -vp 'vehicle/testing/car'
# for ((i=0; i<${#dataset[@]}; i++)); do
for ((i=0; i<600; i++)); do
	if [[ $i -lt $lower_limit ]]; then
		mv -v ${dataset[$i]} 'vehicle/training/car'
	elif [[ $i -gt $upper_limit ]]; then
		mv -v ${dataset[$i]} 'vehicle/validation/car'
	else
		mv -v ${dataset[$i]} 'vehicle/testing/car'
	fi
done