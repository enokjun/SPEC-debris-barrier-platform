# SPEC-debris-barrier-platform

# SPEC-debris-barrier script

use the python file (SPEC_debris_barrier_platform_v8_10.py) to run:

1) SPEC-debris model
	
2) optimal closed and open-type barrier location selection model 
		
3) closed and open-type barrier performance evaluation

to request a software GUI version (refer to SPEC_debris_barrier_platform_GUI.png) and/or user manual (written in Ph.D. Thesis by Enok Cheon),
please send a request through e-mail:
	
	enokjun@gmail.com
	
	enokjun@kaist.ac.kr

Please note that sometimes a mistake is found afterward in SPEC_debris_barrier_platform_v8_10.py;
therefore, try using the version in GitHub instead of the version installed in GUI if there are numerical or computational errors.

# Installation Setup

1) The user must install have installed Python3 on the computer. Head to the website (https://www.python.org/) and install the latest Python 3.9 version

2) The user must contain the following Python libraries; please install these libraries using the pip module:
	
numpy, pandas, laspy, scipy, pykrige, plotly, shapely, matplotlib, alphashape, trimesh, scikit-learn, python-fcl, tripy

These python libraris can be installed by typing the following on the command prompt (or terminal)
	
	python -m pip install numpy pandas laspy scipy pykrige plotly shapely matplotlib alphashape trimesh scikit-learn python-fcl tripy

4) The following 3rd party software are not required but would be helpful:
	
	program code editor (e.g. VS code) with Python extension installed
	
	internet browser (chrome or edge)

# Testing to check whether correctly installed

To run the sample case provided in the Github, please follow these instructions after installation is completed

1) Download all the codes in the GitHub into the computer

2) Open the 'sample_input_JSON.json' on the program code editor (e.g. VS code) 

3) On the 'sample_input_JSON.json' files, go to line 5 with the heading "folder path" and replace the '...' to the full folder path location where the 'sample_case' folder is located, respectively. The name of the folder path should always be between double quotation marks ("  ")

4) Run the 'SPEC_debris_barrier_platform_v8_11.py' to start both simulations. Use one of these methods:

1st method - VS code or IDE

a) open the 'SPEC_debris_barrier_platform_v8_11.py' file in the VS code or IDE

b) run python code

2nd method - terminal/cmd

a) open command prompt or terminal

b) move the current folder to the folder where the 'SPEC_debris_barrier_platform_v8_11.py' file is located

c) type the following code and press enter key:

	python SPEC_debris_barrier_platform_v8_11.py

5) All the simulation files are stored in the folder specified in Step 3, and the results will automatically open on the browsers


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


