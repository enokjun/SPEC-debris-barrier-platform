# SPEC-debris-barrier-platform

# SPEC-debris-barrier script

use the python file (SPEC_debris_barrier_platform_v8_10.py) to run:

1) SPEC-debris model
	
2) optimal closed and open-type barrier location selection model 
		
3) closed and open-type barrier performance evaluation

to request for a software GUI version (refer to SPEC_debris_barrier_platform_GUI.png) and/or user manual (written in PhD Thesis by Enok Cheon),
please send a request through e-mail:
	
	enokjun@gmail.com
	
	enokjun@kaist.ac.kr

Please note that sometimes a mistake is found afterwards in SPEC_debris_barrier_platform_v8_10.py;
therefore, try using the version in GitHub instead of version installed in GUI if numerical or computational errors.

# Installation Setup

1) After installing the exe file, the user must install have installed Python3.
if the user have not installed Python3 yet, please head to the website below and install the latest Python 3.9 version:
https://www.python.org/

2) The user must contain the following Python libraries, please install these libraries using the pip module:
	
numpy, pandas, laspy, scipy, pykrige, plotly, shapely, matplotlib, alphashape, trimesh, scikit-learn, python-fcl, tripy
These python libraris can be installed by typing the following on the command prompt (or terminal)
	
	python -m pip install numpy pandas laspy scipy pykrige plotly shapely matplotlib alphashape trimesh scikit-learn python-fcl tripy

4) The following 3rd party softwares are not required, but would be helpful:
	
	program code editor (e.g. VS code) with python extension installed
	
	internet browser (chrome or edge)

# Tip

Refer to the PhD Thesis "A New Simulation Model for Optimal Location Selection and Performance Evaluation of Barriers as Mitigation Against Debris Flow" by Enok Cheon (2022) as the User manual

ch 2 and 3 - Theory 

ch 4 - Software GUI, features, and instructions

Appendix A - Instruction for creating the input JSON file

# Reference

To reference the software, please use the following:

{HARVARD}
Cheon, E 2022, ‘A New Simulation Model for Optimal Location Selection and Performance Evaluation of Barriers as Mitigation Against Debris Flow’, PhD thesis, Korea Advanced Institute of Science and Technology (KAIST), Daejeon, Republic of Korea.

{APA}
Cheon, E. (2022). A New Simulation Model for Optimal Location Selection and Performance Evaluation of Barriers as Mitigation Against Debris Flow (Doctoral thesis), Korea Advanced Institute of Science and Technology (KAIST), Daejeon, Republic of Korea.


# Example

To run the sample case provided in the Github, please follow these instructions after installation is completed

1) Download the code into the computer

2) Open the 'SPEC_debris_barrier_platform_v8_11.py' and 'sample_input_JSON.json' on the program code editor (e.g. VS code) 

3) On the 'sample_input_JSON.json' file, go to line 5 with heading "folder path". Replace the '...' to the full folder path location where the 'sample_case' folder is located

4) On the 'SPEC_debris_barrier_platform_v8_11.py' file, go to line 19817 where the list named "input_JSON_file_names" is located. Make sure the name of the input JSON files are listed between the square brackets. The name of the input JSON files should be between quatation marks, i.e., string format. 

5) Run the python file to start the simulation. Use one of these methods:

1st method - VS code

a) open to the 'SPEC_debris_barrier_platform_v8_11.py' file in the VS code

b) press F5 button

2nd method - terminal/cmd

a) open command prompt or terminal

b) move the current folder to the folder where the 'SPEC_debris_barrier_platform_v8_11.py' file is located

c) type the following code and press enter key:

	python SPEC_debris_barrier_platform_v8_11.py

6) All the simulation files are stored in the folder specified in step 3

