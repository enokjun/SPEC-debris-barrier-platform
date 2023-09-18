@echo off
rem Install the Python 3.10 version to the computer
if exist "./bin/python-3.10.11-amd64.exe" (
    rem file exists so run the installation file
    echo 64-bit version of Python 3.10 is already installed.
	call "./bin/python-3.10.11-amd64.exe"
) else (
    rem file doesn't exist - download from the internet and setup
    echo 64-bit version of Python 3.10 not found on the computer.
    echo Downloading the installer from python.org and running it...
    start "" https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
)

echo Python 3.10.11 installed on the computer.
pause
echo Now installing the necessary Python libraries to run SPEC-debris-barrier...

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
python -m pip install tripy

echo All necessary Python libraries for SPEC-debris-barrier installed!

