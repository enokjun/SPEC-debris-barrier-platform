#!/usr/bin/python3

'''
Author:     Enok Cheon
Date:       June 15, 2022
Purpose:    SPEC-debris-barrier - GUI
Language:   Python3
License:    MIT

Author:     Enok Cheon 
Date:       June 15, 2022
Purpose:    SPEC-debris-barrier - compiled version
Language:   Python3
License:    

Copyright <2022> <Enok Cheon>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

###########################################################################
## import libraries for GUI
###########################################################################
if __name__ == '__main__':
	print("The programming is importing python libraries for analysis ... ")

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import font

# import threading
import concurrent.futures

import subprocess

import csv

from SPEC_debris_barrier_platform_v8_12 import check_json_input_v8_00

import os
import sys
from platform import system

if __name__ == '__main__':
	print("Importing modules (this may take a while) ... ")


def SPEC_debris_barrier_platform_GUI_v2_00():

	###########################################################################
	## what system are we running on, which text editor is available?
	###########################################################################
	
	opsys = system()
	if not opsys in ["Windows", "Linux", "Darwin"]:
		print("Operating system {:s} not supported.\n".format(opsys))
		sys.exit(1)

	if opsys == "Darwin":
		editor = "TextEdit"				# default editor on macOS
		pdf_viewer = "preview"			# default PDF viewer on macOS
	elif opsys == "Linux":
		flag = 0

		if os.path.isfile("/usr/bin/xdg-open") == True:
			fileman = "xdg-open"
		elif os.path.isfile("/usr/bin/konqueror"):			# KDE
			fileman = "konqueror"
		elif os.path.isfile("/usr/bin/dolphin"):			# KDE
			fileman = "dolphin"
		elif os.path.isfile("/usr/bin/nautilus"):			# Gnome
			fileman = "nautilus"
		elif os.path.isfile("/usr/bin/pcmanfm"):			# e.g. LXDE
			fileman = "pcmanfm"
		elif os.path.isfile("/usr/bin/pcmanfm-qt"):			# e.g. LXqt
			fileman = "pcmanfm-qt"
		else:
			print("\nNo graphical file manager found. Install one among:")
			print("    gedit, kate, sublime_text, geany, leafpad")
			print("or edit the top of SPEC_debris_barrier_GUI_v8_12.py")
			print("to reflect your system's set-up.")
			flag += 1

		if os.path.isfile("/usr/bin/xdg-open"):
			editor = "xdg-open"
		elif os.path.isfile("/usr/bin/gedit") == True:		# Debian & Co.
			editor = "gedit"
		elif os.path.isfile("/usr/bin/kate") == True:		# KDE
			editor = "kate"
		elif os.path.isfile("/usr/bin/sublime_text") == True:
			editor = "sublime_text"
		elif os.path.isfile("/snap/bin/sublime_text") == True:
			editor = "snap run sublime_text"
		elif os.path.isfile("/usr/bin/geany") == True:		# GTK-based distros
			editor = "geany"
		elif os.path.isfile("/snap/bin/geany") == True:		# GTK-based distros
			editor = "snap run geany"
		elif os.path.isfile("/usr/bin/leafpad") == True:	# GTK-based distros
			editor = "leafpad"		
		elif os.path.isfile("/snap/bin/leafpad") == True:	# GTK-based distros
			editor = "snap run leafpad"
		else:
			print("\nNo graphical text editor found. Install one among:")
			print("    gedit, kate, sublime_text, geany, leafpad")
			print("or edit the top of SPEC_debris_barrier_GUI_v8_12.py.")
			flag += 1
		
		if os.path.isfile("/usr/bin/xdg-open") == True:
			pdf_viewer = "xdg-open"
		elif os.path.isfile("/usr/bin/qpdfview") == True:
			pdf_viewer = "qpdfview"
		elif os.path.isfile("/usr/bin/evince") == True:
			pdf_viewer = "evince"
		elif os.path.isfile("/snap/bin/evince") == True:
			pdf_viewer = "snap run evince"
		elif os.path.isfile("/usr/bin/okular") == True:
			pdf_viewer = "okular"
		elif os.path.isfile("/snap/bin/okular") == True:
			pdf_viewer = "snap run okular"
		elif os.path.isfile("/usr/bin/acroread") == True:
			pdf_viewer = "acroread"
		else:
			print("\nNo PDF viewer found. Install one among:")
			print("    qpdfview, okular, evince, acroread")
			print("or edit the top of SPEC_debris_barrier_GUI_v8_12.py.")
			flag += 1

		if flag > 0:
			sys.exit(2)
			
	###########################################################################
	## folder path of where the exe file is installed
	###########################################################################
	if getattr(sys, 'frozen', False):
		application_path = os.path.dirname(sys.executable)
	else:
		try:
			app_full_path = os.path.realpath(__file__)
			application_path = os.path.dirname(app_full_path)
		except NameError:
			application_path = os.getcwd()

	app_path = os.path.join(application_path)

	###########################################################################
	## error code
	###########################################################################
	error_code_dict = {
		0: ['no error', 'no error found'], 
		1: ['JSON file heading error', 'the input file heading has violated the JSON format'], 
		2: ['JSON file variable error', 'the required variable name has a typo or is unavailable'], 
		3: ['python script file naming error', 'the naming of python file has error. Please ensure the python file name follows this format: SPEC_debris_barrier_platform_v00_00.py where 00_00 is the version number'], 
		4: ['python script version error', 'the version number on the python script and the input json file should match'],
		5: ['JSON file name error', 'the input file name has error or is incorrect'], 
		20: ['flowpath file cannot be found', 'the file does not exist or is not located in the folder'],
		21: ['flowpath filetype error', 'must be one of csv or grd or las filetypes'],
		30: ['source file cannot be found', 'the file does not exist or is not located in the folder'], 
		31: ['source filetype error', 'must be one of csv or grd or las filetypes'], 
		40: ['material file cannot be found', 'the file does not exist or is not located in the folder'], 
		41: ['material filetype error', 'must be one of csv or grd or las filetypes'], 
		42: ['material file missing', 'material file required when more than one material is being modelled'], 
		50: ['road_xy file format error', 'check JSON input file format in road xy section'], 
		51: ['road_xy data error', 'two XY coordinates are required to display road'], 
		52: ['road_xy number error', 'two XY coordinates should be expressed with numbers'], 
		53: ['edge_xy points error', 'edge_xy not specified as a following format: [[x1,y2], [x2,y2], ...]'], 
		54: ['edge_xy points data error', 'number of points specified in edge_xy must be greater than two(2)'], 
		55: ['edge_xy points number error', 'integer or floating point number should be specified for edge_xy coordinates'], 
		60: ['time data number error', 'time step interval and maximum simulation time should be expressed as a number'], 
		61: ['t_step value error', 'time step interval should be larger than zero(0)'], 
		62: ['t_max value error', 'maximum simulation time should be larger than zero(0)'], 
		63: ['t_max and t_step inequality error', 't_step < t max'], 
		70: ['particle number number error', 'if particle number is specified the value should be expressed as an integer'], 
		80: ['initial debris flow velocity number error', 'if specified initial velocity should be expressed with numbers'], 
		90: ['material file format error', 'check JSON input file format in material section'], 
		91: ['material density number error', 'density value should be expressed with numbers'], 
		92: ['material density value error', 'density value should exceed zero(0)'], 
		93: ['material phi number error', 'internal friction angle value should be expressed with numbers'], 
		94: ['material phi value error', 'internal friction angle value should range between 0 and 45 degrees'], 
		95: ['material frictional resistance data error', 'frictional resistance fb values should be specified'], 
		96: ['material frictional resistance number error', 'frictional resistance fb value should be expressed as a number higher than or equal to zero(0)'], 
		97: ['material turbulence resistance data error', 'turbulence resistance ft values should be specified'], 
		98: ['material turbulence resistance number error', 'turbulence resistance ft value should be expressed as a number higher than equal to zero(0)'], 
		99: ['material yield stress resistance data error', 'yield stress resistance tau_y values should be specified'], 
		100: ['material yield stress resistance number error', 'yield stress resistance tau_y value should be expressed as a number higher than equal to zero(0)'], 
		101: ['material viscosity based resistance data error', 'viscosity based resistance mu values should be specified'], 
		102: ['material viscosity based resistance number error', 'viscosity based resistance mu value should be expressed as a number higher than equal to zero(0)'], 
		103: ['material entrainment rate number error', 'entrainment growth rate Es value should be expressed with numbers higher or equal to zero (0)'], 
		104: ['maximum erodible depth number error', 'maximum erodible depth value should be expressed with numbers'], 
		105: ['maximum erodible depth value error', 'maximum erodible depth value should be higher or equal to zero(0)'], 
		106: ['Coefficient of restitution with particles number error', 'COR between particles COR_p2p_N and/or COR_p2p_T value should be expressed with numbers'], 
		107: ['Coefficient of restitution with particles value error', 'COR between particles COR_p2p_N and/or COR_p2p_T value must be specified between 0 and 1'], 
		108: ['Coefficient of restitution with wall number error', 'COR between wall COR_p2w_N and/or COR_p2w_T value should be expressed with numbers'], 
		109: ['Coefficient of restitution with wall value error', 'COR between wall COR_p2w_N and/or COR_p2w_T value must be specified between 0 and 1'], 
		900: ['Coefficient of restitution with boundary number error', 'COR between boundary COR_p2b_N and/or COR_p2b_T value should be expressed with numbers'], 
		901: ['Coefficient of restitution with boundary value error', 'COR between boundary COR_p2b_N and/or COR_p2b_T value must be specified between 0 and 1'], 
		903: ['Coefficient of restitution with ground number error', 'COR between ground COR_p2g_N and/or COR_p2g_T value should be expressed with numbers'], 
		904: ['Coefficient of restitution with ground value error', 'COR between ground COR_p2g_N and/or COR_p2g_T value must be specified between 0 and 1'],
		800: ['initial SPH interpolation value error', 'initial SPH interpolation value shoube a boolean (true or false)'], 
		110: ['gravity acceleration number error', 'gravitational acceleration g values should be expressed with numbers'], 
		111: ['gravity unit vector format error', 'gravity vector in XYZ direction should be specified in a list [g_x, g_y, g_z] format'], 
		112: ['gravity unit vector data length error', 'gravity vector component for each XYZ direction should be specified'], 
		113: ['gravity unit vector number error', 'gravity unit vector values should be an integer or floating point number'], 
		114: ['gravity unit vector data magnitude error', 'the magnitude of the gravity unit vector should be equal to one(1)'], 
		120: ['interpolation method string error', 'the specified interpolation method should be expressed as a string among the selected options'], 
		121: ['interpolation method option error', 'the specified interpolation method is not available'], 
		122: ['kriging interpolation semi-variogram option error', 'the specified semi-variogram model is not available for kriging interpolation'], 
		130: ['entrainment model string error', 'the specified entrainment model should be expressed as a string among the selected options'], 
		131: ['entrainment model option error', 'the specified entrainment model is not available'], 
		140: ['boundary algorithm string error', 'the specified boundary algorithm should be expressed as a string among the selected options'], 
		141: ['boundary algorithm option error', 'the specified boundary algorithm is not available'], 
		150: ['Es_theta_var number error', 'Es_theta_var value should be expressed with numbers'], 
		151: ['Es_theta_var value error', 'Es_theta_var value must be specified between 0 and 90'], 
		160: ['local cell size error', 'local cell size should be specified'], 
		161: ['local cell size data error', 'two local cell size values are required'], 
		162: ['local cell size number error', 'local cell size value should be expressed with numbers'], 
		163: ['local cell size value error', 'local cell size value should be positive odd integer'], 
		170: ['l_dp_min number error', 'l_dp_min coefficient for computing minimum SPH smoothing length should be expressed as a number'], 
		171: ['l_dp_min value error', 'l_dp_min coefficient for computing minimum SPH smoothing length should be higher than zero(0)'], 
		172: ['SPH_B_coefficient number error', 'SPH_B_coefficient for computing adaptive SPH smoothing length should be expressed as a number'], 
		173: ['SPH_B_coefficient value error', 'SPH_B_coefficient coefficient for computing adaptive SPH smoothing length should be higher than zero(0)'], 
		180: ['ConcaveHull algorithm parameter format error', 'ConcaveHull algorithm parameter should be in a dictionary format'], 
		181: ['ConcaveHull algorithm parameter key error', 'required concaveHull algorithm parameter is missing'], 
		182: ['ConcaveHull algorithm parameter number error', 'ConcaveHull algorithm parameters should be expressed as numbers'], 
		183: ['ConcaveHull algorithm max iteration value error', 'ConcaveHull algorithm max iteration parameter should be larger than zero(0)'], 
		184: ['ConcaveHull algorithm min alpha value error', 'ConcaveHull algorithm min alpha parameter should be a positive number'], 
		185: ['ConcaveHull algorithm max alpha value error', 'ConcaveHull algorithm max alpha parameter should be a positive number and should be larger than the min alpha parameter'], 
		190: ['merge overlap ratio number error', 'merge overlap ratio should be expressed as a number'], 
		191: ['merge overlap ratio value error', 'merge overlap ratio should be expressed as a positive number between 0 and 1'], 
		210: ['cpu multiprocessing number error', 'number of CPU pool values should be expressed as integer'], 
		211: ['cpu multiprocessing value error', 'number of CPU pool values should be expressed as one(1) or higher positive integer'], 
		300: ['optimal barrier location loop number error', 'optimal barrier location selection process loop value should be expressed as an integer number'], 
		301: ['optimal barrier location limit number error', 'if number of barriers are specified, it should be expressed as an integer'], 
		302: ['optimal barrier location limit value error', 'if number of barriers are specified it should be expressed as a positive integer and the maximum barrier number should not exceed the minimum barrier number'], 
		303: ['optimal barrier location avoid DFL option error', 'check opt_avoid_dfl (true/false) options'], 
		310: ['optimal closed barrier number error', 'the optimization weighting factors should be expressed as numbers and the maximum barrier number should not exceed the minimum barrier number'], 
		311: ['optimal closed barrier sum value error', 'the sum of optimization weighting factors should add up to 1'], 
		320: ['optimal combined barrier number error', 'the optimization weighting factors should be expressed as numbers'], 
		321: ['optimal combined barrier sum value error', 'the sum of optimization weighting factors should add up to 1'], 
		330: ['open type barrier number error', 'the performance ratio of open type barriers should be expressed as numbers'], 
		331: ['open type barrier SR value error', 'the speed ratio (SR) should be between 0 and 1'], 
		332: ['open type barrier TR value error', 'the trap ratio (TR) should be between 0 and 1'], 
		340: ['vulnerability analysis number error', 'the data used for vulnerability analysis is incorrect'], 
		341: ['vulnerability analysis min_uV value error', 'the minimum speed and volume for vulnerability analysis should be a positive number'], 
		342: ['vulnerability analysis VI_crit value error', 'the vulnerability index (VI) should be between 0 and 1'], 
		350: ['output_optimal_step option error', 'check output_optimal_step output options'], 
		400: ['figure option error', 'check plotting (true/false) options'], 
		410: ['2D plot option error', 'check whether there are nine(9) 2D plot legend maximum values'],
		411: ['2D plot option value error', 'check whether 2D plot legend maximum values are zero(0) or a negative number'], 
		420: ['3D plot option error', 'check 3D plot legend maximum or Z offet values'], 
		421: ['3D plot option value error', 'check whether 3D plot legend maximum or Z offet values are zero(0) or a negative number'], 
		430: ['animation option error', 'check 2D and 3D animation plot options'], 
		431: ['animation option value error', 'check whether 2D and 3D animation option values are zero(0) or a negative number'], 
		440: ['figure size error', 'check size of the following: marker, line, layout width, layout height'], 
		441: ['figure size value error', 'check size of the following: marker, line, layout width, layout height'], 
		500: ['csv output option error', 'check csv output options'], 
		510: ['decimal point (dp) number error', 'decimal point (dp) values should be expressed as integer'], 
		511: ['decimal point (dp) value error', 'decimal point (dp) values should be expressed as zero or higher positive integer'], 
		600: ['wall data error', 'the wall/building data is not specified'], 
		601: ['wall type option error', 'the wall/building type is not available'], 
		610: ['slit-type wall parameter number error', 'slit-type wall parameters should be expressed as a number'], 
		611: ['slit-type wall: slit ratio value error', 'slit-type wall slit ratio parameter should be between 0 and 1'], 
		612: ['slit-type wall: number of wall segments value error', 'slit-type wall number of wall segments should be an integer larger than 0'], 
		613: ['slit-type wall: orientation angles value error', 'slit-type wall orientations should be an polar angle degree between and including -90 and 90'],
		614: ['slit-type wall: thickness value error', 'slit-type wall thickness should be larger than zero(0)'], 
		615: ['slit-type wall: length value error', 'slit-type wall length should be larger than zero(0)'], 
		616: ['slit-type wall: thickness and length inequality error', 'slit-type wall length >= slit-type wall thickness'], 
		617: ['slit-type wall: elevation option error', 'the selected wall elevation option is unavailable'], 
		618: ['slit-type wall: elevation or height value error', 'the elevation or height value should be larger than zero(0)'],
		620: ['circular baffles parameter number error', 'circular baffle wall parameters should be expressed as a number'], 
		621: ['circular baffles: number of cylinder value error', 'circular baffle wall number of wall segments should be an integer larger than 1'], 
		622: ['circular baffles: orientation angles value error', 'circular baffle wall orientations should be an polar angle degree between and including -90 and 90'], 
		623: ['circular baffles: radius value error', 'circular baffle radius should be larger than zero(0)'], 
		624: ['circular baffles: section length value error', 'circular baffle wall length should be larger than zero(0)'], 
		625: ['circular baffles: thickness and length inequality error', 'circular baffle wall length >= circular baffle wall thickness (=2*radius)'], 
		626: ['circular baffles: elevation option error', 'the selected wall elevation option is unavailable'], 
		627: ['circular baffles: elevation or height value error', 'the elevation or height value should be larger than zero(0)'], 
		630: ['building parameter number error', 'building parameters should be expressed as a number'], 
		631: ['building: elevation option error', 'the selected building elevation option is unavailable'], 
		632: ['building: elevation or height value error', 'the elevation or height value should be larger than zero(0)'], 
		633: ['building: XY points data number error', 'the number of building XY points should be larger than two(2)'], 
		634: ['building: XY points number error', 'building XY coordinates should be expressed as a number'], 
		635: ['building: XYZ points data number error', 'the number of building XYZ points should be larger than two(2)'], 
		636: ['building: XYZ points number error', 'building XYZ coordinates should be expressed as a number'], 
		637: ['ditch parameter number error', 'ditch parameters should be expressed as a number'], 
		638: ['ditch: elevation option error', 'the selected ditch elevation option is unavailable'], 
		639: ['ditch: elevation or height value error', 'the elevation or height value should be less than zero(0)'], 
		640: ['ditch: XY points data number error', 'the number of ditch XY points should be larger than two(2)'], 
		641: ['ditch: XY points number error', 'ditch XY coordinates should be expressed as a number'], 
		642: ['ditch: XYZ points data number error', 'the number of ditch XYZ points should be larger than two(2)'], 
		643: ['ditch: XYZ points number error', 'ditch XYZ coordinates should be expressed as a number'], 
		700: ['equivalent basal resistance number error', 'equivalent basal resistance should be expressed as a number'], 
		701: ['equivalent basal resistance value error', 'equivalent basal resistance should be 0 or positive value'], 
		810: ['part_input file error', 'the initial particle data could not be found'], 
		811: ['part_input filetype error', 'the initial particle data should be in CSV file format'], 
		999: ['unknown error', 'unidentified error, please check input once more']
	}

	###########################################################################
	## set up GUI root
	###########################################################################
	root = Tk()

	root.title("SPEC-debris-barrier")    # S/W title
	# PNG should work on all three platforms. I converted GUI_GEL_icon.ico to
	# PNG format and uncommented the following line.
	root.call('wm', 'iconphoto', root._w, PhotoImage(file='GUI_GEL_icon.png'))     # icon
	# The following line does not work on Linux (and presumably not on macOS):
	#root.iconbitmap(app_path+'/GUI_GEL_icon.ico')    # icon
	# root.geometry("460x200")            # window size

	###########################################################################
	## label and input text box
	###########################################################################
	# display input file name - or user can directly type in the name
	SPEC_label = Label(root, text="SPEC-debris-barrier", font=("Arial", 16)) 
	creator_label = Label(root, text="Created by Enok Cheon", font=("Arial", 12)) 
	version_label = Label(root, text="ver 1.20", font=("Arial", 10)) 

	# display input file name - or user can directly type in the name
	input_label = Label(root, text="INPUT", font=("Arial", 12)) 
	open_file_location_name = Entry(root, width=40, bd=3, font=("Arial", 12)) 

	# initial status 
	status = Label(root, text="", bd=1, relief=SUNKEN, anchor=E, font=("Arial", 12))

	###########################################################################
	## commands to operate when button is clicked
	###########################################################################
	# open button - open input file name and display it on the open_file_location_name entry
	def open_command():

		## reset to initial condition
		# disable reading, folder and checking buttons when new input file is being opened
		read_button.config(state = DISABLED)
		folder_button.config(state = DISABLED)
		check_button.config(state = DISABLED)
		run_button.config(state = DISABLED)

		# erase all status text
		try:
			if len(root.filename) > 0:
				status.config(text="selecting new input JSON file ")
		except:
			status.config(text="")

		# delete whatever is written in the file name text input box
		open_file_location_name.delete(0, END)

		# if file loading was done previously once, then reopen the current folder 
		if len(status.cget("text")) > 0:
			file_path_name = root.filename
			file_naming_list = file_path_name.split("/")
			file_path = "/".join(file_naming_list[:-1])

		# if brand new, then by default open document
		else: 
			file_path = "C:/Users/"+os.getlogin()+'/Documents'
		
		# load input file
		root.filename = filedialog.askopenfilename(initialdir=file_path, 
													title="Select input file",
													# filetypes=(("json files", "*.json"), ("all files", "*.*"))
													filetypes=(("json files", "*.json"),)   # only open json files
													)
		
		# open input json file and display file location in the input text
		try:
			if len(root.filename) > 0:
				file_path_name = root.filename
				file_naming_list = file_path_name.split("/")
				file_name_only = file_naming_list[-1]

				open_file_location_name.insert(0, file_name_only)
			
				# enable reading, folder and checking buttons
				read_button.config(state = NORMAL)
				folder_button.config(state = NORMAL)
				check_button.config(state = NORMAL)
				# run_button.config(state = NORMAL)

				# update status to show similuation is running 
				status.config(text = "input JSON file loaded ")
		except:
			pass

		return None

	# open the input json file and read
	def read_command():
		# open the json input file
		if system() == "Windows":
			os.startfile(root.filename)
			return None
		elif system() == "Linux":
			subprocess.run([editor, root.filename])
		elif system() == "Darwin":
			subprocess.run(["open", root.filename, "-a", editor])

	# open a folder for selecting a file
	def folder_command():
		# extract only the folder path text
		file_path_name = root.filename
		file_naming_list = file_path_name.split("/")
		file_path = "/".join(file_naming_list[:-1])

		# open the folder path
		if opsys == "Windows":
			os.startfile(file_path+'/')
		elif system() == "Linux":
			subprocess.run([fileman, file_path+'/'])
		elif system() == "Darwin":
			subprocess.run(["open", file_path])

		return None

	# open the pdf with user manual
	def help_command():
		# name of help file
		help_file = app_path+"/help/PhD Thesis Enok Cheon KAIST.pdf"
		if opsys == "Windows":
			os.startfile(help_file)
		elif opsys == "Darwin":
			subprocess.run(["open", help_file])
		else:
			subprocess.run(["xdg-open", help_file])
		# webbrowser.open_new("https://www.google.com")
		return None

	# run checking for error in input json file for SPEC-debris-barrier
	def check_command():

		global check_result
		check_result = None

		# update status to show simulation is running 
		status.config(text = "checking JSON input ")

		#with concurrent.futures.ThreadPoolExecutor() as executor:
		#	checking = executor.submit(check_json_input_v8_00, root.filename)
		#	check_result = checking.result()
		check_result = 0
		
		# update status to show simulation is running 
		status.config(text = "check complete ")
		
		# pop up message based on simulation results
		if check_result == 0:       # no error
			run_button.config(state = NORMAL)   # enable to run simulation after checking shows no error
			proceed = messagebox.askyesno("Check Result", "No error found in input file.\n\nDo you want to proceed with the simulation?")
			if proceed:
				run_command()

#DI		These lines look strange: If there is something wrong with the input,
#DI		check_result returns a tuple, not a large number!		
		elif check_result == 1 or check_result == 2:
			# get only simulation results up to hundredth number or below
			messagebox.showerror("Error Code "+str(check_result), "Error Code "+str(check_result)+"\n\n"+error_code_dict[check_result]["error name"]+':\n\n'+error_code_dict[check_result]["error description"])

		else:
			# get only simulation results up to hundredth number or below
			error_result = int(check_result - round(check_result, ndigits=-3))
			json_nth_number = int(round(check_result, -3)/1000)

			messagebox.showerror("Error Code "+str(check_result), "Error Code "+str(error_result)+" for No."+str(json_nth_number)+" Simulation Queue\n\n"+error_code_dict[error_result]["error name"]+':\n\n'+error_code_dict[error_result]["error description"])

		return None

	# run SPEC-debris-barrier and display message for completion/error
	def run_command():

		if check_result == 0:
			run_button.config(state = NORMAL)
		else:
			run_button.config(state = DISABLED)

		# update status to show similuation is running 
		status.config(text = "simulation running ")

		# extract only the folder path text
		file_path_name = root.filename
		file_naming_list = file_path_name.split("/")
		file_path = "/".join(file_naming_list[:-1])

		file_name_ext_list = file_naming_list[-1].split(".")
		file_name = ".".join(file_name_ext_list[:-1])
		# file_ext = file_name_ext_list[-1]

		run_python_code = '\"'+app_path+'/SPEC_debris_barrier_platform_v8_12.py\" \"'+file_path_name+'\"' 

		## create a bat file to run simulation on cmd
		if opsys == "Windows":
			# bat_code =	"@echo off\npython " \
			# 			+ run_python_code \
			# 			+ " " + '\"' + file_path_name + '\"' \
			# 			+ "\npause\nexit"

			# with open(file_path+"/"+file_name+".bat", mode="wt") as f:
			# 	f.write(bat_code)

			# proc = subprocess.Popen(file_path+"/"+file_name+".bat", shell=False,
			# 						stdout=sys.stdout, stderr=subprocess.PIPE,
			# 						encoding='ascii')

			# if osGEO4w_run_response:   # run on osGeo4w python package
			# 	popen_cmd_code = "C:/OSGeo4W/apps/Python39/python3.exe "+run_python_code+" && pause && exit"
			# else:

			popen_cmd_code = "python "+run_python_code+" && pause && exit"
			proc = subprocess.Popen(popen_cmd_code, shell=False, creationflags=subprocess.CREATE_NEW_CONSOLE, stdout=sys.stdout, stderr=subprocess.PIPE, encoding='ascii')

		elif system() == "Linux": # Linux use #!/usr/bin/python3						
			proc = subprocess.Popen(
					[ "python3",
					  app_path+'/SPEC_debris_barrier_platform_v8_12.py',
					  file_path_name ],
					shell=False,	stdout=sys.stdout,
					stderr=subprocess.PIPE, encoding='ascii')
		
		elif system() == "Darwin":

			# bash_code =	"#!/bin/bash\npython3 " \
			# 			+ run_python_code \
			# 			+ " " + '\"' + file_path_name + '\"' \
			# 			+ "\nexit"

			# with open(file_path+"/"+file_name+".sh", mode="wt") as f:
			# 	f.write(bash_code)
			# f.close()

			# proc = subprocess.Popen(["sh", file_path+"/"+file_name+".sh"], shell=False,
			# 						stdout=sys.stdout, stderr=subprocess.PIPE,
			# 						encoding='ascii')

			proc = subprocess.Popen(
					[ "python3",
					  app_path+'/SPEC_debris_barrier_platform_v8_12.py',
					  file_path_name ],
					shell=False,	stdout=sys.stdout, 
					stderr=subprocess.PIPE, encoding='ascii')
		
		while proc.poll() is None:
			pass

		proc.kill()     # close cmd
		errX = proc.communicate()[1]  # export error message
		print(errX)

		# update status to show simulation is complete
		status.config(text = "simulation complete ")
		
		if errX == '':     # no error
			messagebox.showinfo("Simulation Result", "Completed the simulation")
		else:     # some sort of error occurred
			messagebox.showerror("Simulation Result", "Error 999: unknown error has occurred")

		return None

	# stop currently operating SPEC-debris-barrier
	def quit_command():

		if status.cget("text") == "simulation running":
		
			quit_response = messagebox.askyesno("Quit Simulation", "Are you sure you want to quit the simulation and close?")

			# pop up message based on simulation results
			if quit_response:       # cancel simulation
				root.quit()
		
		else:
			root.quit()

		return None

	###########################################################################
	## button widget
	###########################################################################
	## create a button widget
	# open input file folderpath and filename
	open_button = Button(root, text="Open", width=4, height=1, padx=10, pady=5, font=("Arial", 12), command=open_command) 

	# read the input file by opening them
	read_button = Button(root, text="Read", width=4, height=1, padx=10, pady=5, font=("Arial", 12), command=read_command, state=DISABLED)

	# open the json folder in folder nagivation
	folder_button = Button(root, text="Folder", width=4, height=1, padx=10, pady=5, font=("Arial", 12), command=folder_command, state=DISABLED)

	# check json input
	# check_button = Button(root, text="Check", width=4, height=1, padx=10, pady=5, font=("Arial", 12), command=lambda: threading.Thread(target=check_command(open_file_location_name.get())).start())
	check_button = Button(root, text="Check", width=4, height=1, padx=10, pady=5, font=("Arial", 12), command=check_command, state=DISABLED)

	# execute simulation
	# run_button = Button(root, text="Run", width=4, height=1, padx=10, pady=5, font=("Arial", 12), command=lambda: threading.Thread(target=run_command(open_file_location_name.get())).start())
	run_button = Button(root, text="Run", width=4, height=1, padx=10, pady=5, font=("Arial", 12), command=run_command, state=DISABLED)

	# help button
	help_button = Button(root, text="Help", width=4, height=1, padx=10, pady=5, font=("Arial", 12), command=help_command)

	###########################################################################
	## placement of widgets on GUI
	###########################################################################
	# title and version
	SPEC_label.grid(row=0, column=0, columnspan=6, padx=10, pady=(10,5), sticky=W+E)
	creator_label.grid(row=1, column=0, columnspan=6, padx=10, sticky=W+E)
	version_label.grid(row=2, column=0, columnspan=6, padx=10, pady=3, sticky=W+E)

	# input file
	input_label.grid(row=3, column=0, padx=10, pady=(0,5))
	open_file_location_name.grid(row=3, column=1, columnspan=5, padx=10)
	# open_file_location_name.grid(row=3, column=0, columnspan=6, sticky=W+E, padx=10)

	# buttons
	help_button.grid(row=4, column=0, padx=5, pady=5)
	open_button.grid(row=4, column=1, padx=5, pady=5)
	read_button.grid(row=4, column=2, padx=5, pady=5)
	folder_button.grid(row=4, column=3, padx=5, pady=5)
	check_button.grid(row=4, column=4, padx=5, pady=5)
	run_button.grid(row=4, column=5, padx=5, pady=5)
	
	# status
	status.grid(row=6, column=0, columnspan=6, sticky=W+E)

	###########################################################################
	## GUI operation
	###########################################################################
	# check if the closing the window is valid
	root.protocol("WM_DELETE_WINDOW", quit_command)

	# window resize is disabled
	root.resizable(width=False, height=False)

	# run GUI
	root.mainloop()

if __name__ == '__main__':
	SPEC_debris_barrier_platform_GUI_v2_00()
