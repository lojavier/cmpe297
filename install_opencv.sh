#!/bin/bash

OPENCV_VERSION="4.0.1"
cd ~/
BASHRC_FILE="${HOME}/.bashrc"

upgrade_packages_1 () {
	sudo apt update --fix-missing && sudo apt upgrade -y
	return $?
}

upgrade_packages_2 () {
	sudo apt-get update --fix-missing && sudo apt-get upgrade -y
	return $?
}

install_packages () {
	echo "$1"
	sudo apt-get update --fix-missing && sudo apt-get install -y "$1"
	return $?
}

if [[ $1 -eq 1 ]]; then

	ret=-1
	while [ $ret -ne 0 ]; do
		ret=$(upgrade_packages_1)
		echo "$ret"
	done
	ret=-1
	while [ $ret -ne 0 ]; do
		ret=$(upgrade_packages_2)
		echo "$ret"
	done
	ret=-1
	while [ $ret -ne 0 ]; do
		ret=$(install_packages "build-essential cmake unzip pkg-config git")
		echo "$ret"
	done
	ret=-1
	while [ $ret -ne 0 ]; do
		ret=$(install_packages "libjpeg-dev libpng-dev libtiff-dev")
		echo "$ret"
	done
	ret=-1
	while [ $ret -ne 0 ]; do
		ret=$(install_packages "libxvidcore-dev libx264-dev")
		echo "$ret"
	done
	ret=-1
	while [ $ret -ne 0 ]; do
		ret=$(install_packages "libgtk-3-dev")
		echo "$ret"
	done
	ret=-1
	while [ $ret -ne 0 ]; do
		ret=$(install_packages "libatlas-base-dev gfortran")
		echo "$ret"
	done
	ret=-1
	while [ $ret -ne 0 ]; do
		ret=$(install_packages "python3-dev python3-pip")
		echo "$ret"
	done

	sudo apt-get clean -y
	sudo apt-get autoremove -y

	cd ~/
	wget -O opencv.zip "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip"
	wget -O opencv_contrib.zip "https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip"

	unzip opencv.zip
	unzip opencv_contrib.zip

	mv "opencv-${OPENCV_VERSION}" opencv
	mv "opencv_contrib-${OPENCV_VERSION}" opencv_contrib

	cd ~/
	wget https://bootstrap.pypa.io/get-pip.py
	sudo python3 get-pip.py
	sudo pip install virtualenv virtualenvwrapper
	sudo rm -rf ~/get-pip.py ~/.cache/pip
	pip install --upgrade pip
	pip install --upgrade setuptools
	pip install --upgrade wheel
	# pip install --upgrade setuptools --ignore-installed

	echo -e "\n# virtualenv and virtualenvwrapper" >> $BASHRC_FILE
	echo "export WORKON_HOME=$HOME/.virtualenvs" >> $BASHRC_FILE
	echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> $BASHRC_FILE
	echo "source /usr/local/bin/virtualenvwrapper.sh" >> $BASHRC_FILE
	source $BASHRC_FILE
	export WORKON_HOME=$HOME/.virtualenvs
	export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
	source /usr/local/bin/virtualenvwrapper.sh

	mkvirtualenv cv -p python3

elif [[ $1 -eq 2 ]]; then
# [[ "$VIRTUAL_ENV" == "" ]]; INVENV=$?
# if [[ "$INVENV" == "1" ]]; then
	
	# workon cv

	pip install --upgrade pip
	pip install --upgrade setuptools
	pip install --upgrade wheel 
	pip install --upgrade scipy 
	pip install --upgrade matplotlib 
	pip install --upgrade scikit-image 
	pip install --upgrade scikit-learn 
	pip install --upgrade ipython
	pip install --upgrade dlib
	pip install --upgrade numpy
	pip install --upgrade imutils
	pip install --upgrade socketio
	pip install --upgrade aiohttp
	pip install --upgrade eventlet
	pip install --upgrade pillow
	pip install --upgrade flask
	pip install --upgrade keras
	pip install --upgrade tensorflow

	cd ~/opencv/
	mkdir -p build
	cd build/
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D INSTALL_PYTHON_EXAMPLES=ON \
		-D INSTALL_C_EXAMPLES=OFF \
		-D OPENCV_ENABLE_NONFREE=ON \
		-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
		-D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python3 \
		-D BUILD_EXAMPLES=ON ..

	make -j4

elif [[ $1 -eq 3 ]]; then

	sudo make install
	sudo ldconfig

	cd ~/
	ls -l /usr/local/lib/python3.6/site-packages/cv2/python-3.6/
	
	cd /usr/local/lib/python3.6/site-packages/cv2/python-3.6/
	sudo mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so

	cd ~/.virtualenvs/cv/lib/python3.6/site-packages/
	ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.so cv2.so

	deactivate

	cd ~/

fi