#!/bin/bash

# check if Python3 is installed in Mac
type -p python3 >/dev/null 2>& open https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg && echo Python 3 is already installed

# check pip is installed in Mac
type -p pip3 --version >/dev/null 

echo Now installing the necessary Python libraries to run SPEC-debris-barrier...

python3 -m pip install pip --upgrade
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
python3 -m pip install threadpoolctl==3.1.0

echo All necessary Python libraries for SPEC-debris-barrier installed!

