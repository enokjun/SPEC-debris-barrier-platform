#!/bin/bash

# check if Python3 is installed in Linux
type -p python3 >/dev/null 2>& sudo apt install python3.10 && echo Python 3 is already installed

# check pip is installed in linux
type -p pip3 --version >/dev/null 

echo Now installing the necessary Python libraries to run SPEC-debris-barrier...

python3 -m pip install --upgrade pip
python3 -m pip install numpy
python3 -m pip install pandas
python3 -m pip install scipy
python3 -m pip install plotly
python3 -m pip install matplotlib
python3 -m pip install shapely
python3 -m pip install laspy
python3 -m pip install pykrige
python3 -m pip install alphashape
python3 -m pip install trimesh
python3 -m pip install scikit-learn
python3 -m pip install python-fcl
python3 -m pip install tripy

echo All necessary Python libraries for SPEC-debris-barrier installed!

