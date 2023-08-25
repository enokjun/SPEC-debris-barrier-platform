# SPEC-debris-barrier-platform

# SPEC-debris-barrier script

use the python file (SPEC_debris_barrier_platform_v8_12.py) to run:

1) SPEC-debris model
	
2) optimal closed and open-type barrier location selection model 
		
3) closed and open-type barrier performance evaluation

please send a request through e-mail for any further questions:
	
	enokjun@gmail.com
	
	enokjun@kaist.ac.kr

Please note that sometimes a mistake is found afterward in SPEC_debris_barrier_platform_v8_12.py;
Therefore, try using the version in GitHub instead of the version installed in GUI if there are numerical or computational errors.

# GUI Installation Instructions (Windows)

1) Download the GUI version of the SPEC-debris-barrier platform, which is contained in the "SPEC-debris-barrier-platfrom-GUI-windows" folder

2) Open the "Instructions and Guide for SPEC-debris-barrier Platform GUI.pdf" in the folder "SPEC-debris-barrier-platfrom-GUI-windows/help/" and follow the instructions

# Testing the SPEC-debris-barrier-platform GUI (Windows)

To run the sample cases provided in the Github, please follow these instructions after installation is completed

1) Download all the sample cases "sample_case" and "sample_case_barrier" in the GitHub into the computer

2) Run the "SPEC_debris_barrier_GUI.exe" and follow the instructions of "Instructions for Running Analyses" written in the "Instructions and Guide for SPEC-debris-barrier Platform GUI.pdf"

3) Open one of the sample cases ("sample_case" and "sample_case_barrier") to run


# Python Code Installation Instructions

1) The user must have installed Python (3.9 or 3.10) contain the following Python libraries; please install these libraries using the pip module:
	
numpy, pandas, laspy, scipy, pykrige, plotly, shapely, matplotlib, alphashape, trimesh, scikit-learn, python-fcl, tripy

These python libraris can be installed by typing the following on the command prompt (or terminal)

For windows command prompt:
	
	python -m pip install numpy pandas laspy scipy pykrige plotly shapely matplotlib alphashape trimesh scikit-learn python-fcl tripy

For MacOS terminal:

 	python3 -m ensurepip
  
	python3 -m pip3 install numpy pandas laspy scipy pykrige plotly shapely matplotlib alphashape trimesh scikit-learn python-fcl tripy

For Linux terminal:
	
	sudo apt-get install python3-pip python-dev
 
	pip3 install numpy pandas laspy scipy pykrige plotly shapely matplotlib alphashape trimesh scikit-learn python-fcl tripy

2) The following 3rd party software are not required but would be helpful:
	
	program code editor (e.g. VS code) with Python extension installed
	
	internet browser (chrome or edge)

# Testing to check whether Python Code Runs Correctly

To run the sample case provided in the Github, please follow these instructions after installation is completed

1) Download all the sample cases "sample_case" and "sample_case_barrier" in the GitHub into the computer

2) Open the 'sample_input_JSON.json' on the program code editor (e.g. VS code) 

3) On the 'sample_input_JSON.json' files, go to line 5 with the heading "folder path" and replace the '...' to the full folder path location where the 'sample_case' folder is located, respectively. The name of the folder path should always be between double quotation marks ("  ")

4) Run the 'SPEC_debris_barrier_platform_v8_12.py' located in the folder "SPEC-debris-barrier-platfrom-GUI-windows/bin/" to start both simulations. Use one of these methods:

Terminal/Cmd

a) open command prompt or terminal

b) move the current folder to the folder where the 'SPEC_debris_barrier_platform_v8_12.py' file is located

c) type the following code and press enter key:

	python SPEC_debris_barrier_platform_v8_12.py

OSGeo4W - (download and install from https://trac.osgeo.org/osgeo4w/; then install all the required Python libraries with pip)

a) open OSGeo4W shell located in Start Menu or Desktop 

b) move the current folder to the folder where the 'SPEC_debris_barrier_platform_v8_12.py' file is located

c) type the following code and press enter key:

	python3 SPEC_debris_barrier_platform_v8_12.py <add folder path to where sample case folders>/sample_case/sample_input_JSON.json
 
	python3 SPEC_debris_barrier_platform_v8_12.py <add folder path to where sample case folders>/sample_case_barrier/sample_input_wall_JSON.json

5) All the simulation files are stored in the folders containing the JSON inputs, and the results will automatically open on the browsers

# Reference and Guide

Refer to the Ph.D. Thesis "A New Simulation Model for Optimal Location Selection and Performance Evaluation of Barriers as Mitigation Against Debris Flow" by Enok Cheon (2022) as the User manual

ch 2 and 3 - Theory 

ch 4 - Software GUI, features, and instructions

Appendix A - Instruction for creating the input JSON file

# Bibliography

To reference the software, please use the following:

{HARVARD}
Cheon, E 2022, ‘A New Simulation Model for Optimal Location Selection and Performance Evaluation of Barriers as Mitigation Against Debris Flow’, Ph.D. thesis, Korea Advanced Institute of Science and Technology (KAIST), Daejeon, Republic of Korea.

{APA}
Cheon, E. (2022). A New Simulation Model for Optimal Location Selection and Performance Evaluation of Barriers as Mitigation Against Debris Flow (Doctoral thesis), Korea Advanced Institute of Science and Technology (KAIST), Daejeon, Republic of Korea.


