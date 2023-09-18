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

# GUI Installation Instructions

1) Download the entire github repository codes.
   	(a) Press the "<> Code" button
   	(b) Press the "Download ZIP"
	(c) Save the ZIP file into any folder of your choice, then extract the files in the ZIP file

2) Start the automatic installation process.
   
   (Windows) double-click the "Python_Install_and_Setup_Windows.bat" file in the "SPEC-debris-barrier" folder and follow the instructions.

   (Linux or Mac) follow these steps:
   	(a) open the terminal
   	(b) navigate to your terminal into the "SPEC-debris-barrier" folder by typing the following command:
   
   		cd <filepath of the folder containing the "SPEC-debris-barrier" folder>/SPEC-debris-barrier

   	(c) copy and type the following command to start the installation process.

   	For Linux:
	```
	sh Python_Install_and_Setup_Linux.sh
	```

	For Mac: 

	```
	sh Python_Install_and_Setup_Mac.sh
	```

   NOTE: if Python3 is not already installed on the PC, the installation process will prompt the user to download and install the Python3 installation file. Please refer to the "Instructions and Guide for SPEC-debris-barrier Platform GUI.pdf" located in the "SPEC-debris-barrier/help/" folder when installing Python3

# SPEC-debris-barrier-platform GUI Instructions

The buttons in the GUI perform the following functions:

![SPEC_debris_barrier_platform_GUI](https://github.com/enokjun/SPEC-debris-barrier-platform/assets/11845689/93f19f12-ef8d-4ccb-956e-3321979b15f5)

In Linux and Mac, the terminal will run instead of the command prompt (cmd)

# Test case the SPEC-debris-barrier-platform GUI

To run the sample cases provided in the Github, please follow these instructions after installation is completed.

(windows)
1) double-click "SPEC_debris_barrier_GUI_windows.bat" and wait for the GUI to open

2) press the "Open" button on the GUI. Navigate and open the "sample_input_JSON.json" in "sample_case" folder or "sample_input_wall_JSON.json" in "sample_case_barrier" folder

3) Press the "Check" button to see if any error is found in the input JSON files

4) If no error is found, a message window will ask whether to start the analysis. If the "Yes" button is selected, the simulation begins.
   If the "No" button is pressed, the simulation does not proceed. The user can manually start the simulation by clicking the "Run" button.
   
5) All the simulation files are stored in the folders containing the JSON inputs, and the results will automatically open on the browsers

(Linux or Mac)
1) open the terminal

2) navigate to your terminal into the "SPEC-debris-barrier" folder by typing the following command:
   ```
   cd <filepath of the folder containing the "SPEC-debris-barrier" folder>/SPEC-debris-barrier
   ```
   
3) copy and type the following command to start the GUI (above for Linux and bottom for Mac) and wait for the GUI to open:

    For Linux:
    ```
    sh SPEC_debris_barrier_GUI_linux.sh
    ```

    For Mac: 
    ```
    sh SPEC_debris_barrier_GUI_mac.sh
    ```

5) press the "Open" button on the GUI. Navigate and open the "sample_input_JSON.json" in "sample_case" folder or "sample_input_wall_JSON.json" in "sample_case_barrier" folder

6) Press the "Check" button to see if any error is found in the input JSON files

7) If no error is found, a message window will ask whether to start the analysis. If the "Yes" button is selected, the simulation begins.
   If the "No" button is pressed, the simulation does not proceed. The user can manually start the simulation by clicking the "Run" button.
   
8) All the simulation files are stored in the folders containing the JSON inputs, and the results will automatically open on the browsers

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


