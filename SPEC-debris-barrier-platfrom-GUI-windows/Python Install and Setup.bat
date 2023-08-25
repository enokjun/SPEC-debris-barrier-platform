@echo off
rem Install the Python 3.10 version to the computer
call "./bin/python-3.10.11-amd64.exe"

rem installing necessary python libraries to run SPEC-debris-barrier
python -m pip install pip --upgrade
python -m pip install numpy
python -m pip install pandas
python -m pip install scipy
python -m pip install plotly
python -m pip install matplotlib
python -m pip install shapely
python -m pip install laspy
python -m pip install pykrige
python -m pip install alphashape
python -m pip install trimesh
python -m pip install scikit-learn
python -m pip install python-fcl
