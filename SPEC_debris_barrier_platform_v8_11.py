'''
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

#################################################################################################################
## import library that all analysis relies on
#################################################################################################################

## general purpose
import numpy as np
import time
import math
from datetime import datetime, timedelta
import decimal
import pandas as pd
from copy import deepcopy
import json
import itertools
import os
import sys 
import warnings

# import json file into class
from types import SimpleNamespace

## multiprocessing
import multiprocessing as mp
# import concurrent.futures
# import subprocess

## import geoFileConvert
import csv
import laspy
from scipy.stats import mode

## interpolation functions
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D

## plotly plotting
from plotly.offline import plot
import plotly.graph_objs as go

## shapely
from shapely.geometry import Point, MultiPoint, LineString, Polygon, MultiLineString, MultiPolygon, box
from shapely.ops import polygonize, unary_union, cascaded_union
from shapely.affinity import rotate
from shapely.errors import ShapelyDeprecationWarning

## matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.path import Path

from matplotlib.ticker import MaxNLocator

## Concave-Hull - check_merge_CCH
from scipy.spatial import Delaunay
import alphashape
import trimesh

## Convex-Hull
from scipy.spatial import ConvexHull

## Kmeans - setup at time step = 0 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

## discritize - setup at time step = 0 
import random
from scipy.spatial import Voronoi

## collision - vector and matrix
from scipy.spatial import KDTree
from scipy.linalg import block_diag, inv

## successor - debris-flow particles
from sklearn.linear_model import LinearRegression
from scipy.special import comb

## collision
from scipy.stats import rankdata

## others
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean

# exclude outlier debris-flow particle positions
from sklearn.neighbors import LocalOutlierFactor

## contact / collision
import fcl
import tripy

### shapely warning message ignore
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

###########################################################################
### convert from csv or txt to list
###########################################################################
def csv2list(fileName, starting_row=0):
	# import csv
	with open(fileName, 'r') as f:
		reader = csv.reader(f)
		csvListTxt = list(reader)

	csvListNum = []
	for idR in range(starting_row, len(csvListTxt)):	
		csvListTxt[idR] = [x for x in csvListTxt[idR] if x != '']
		tempList = [float(i) for i in csvListTxt[idR]]
		csvListNum.append(tempList)

	return csvListNum

def txt2list(fileName, starting_row=0):
	with open(fileName, 'r') as myfile:
		data=myfile.read().split('\n')

	txtListNum = []
	for idR in range(starting_row, len(data)):
		tempList1 = data[idR].split('\t') 
		tempList2 = [float(i) for i in tempList1]
		txtListNum.append(tempList2)

	return txtListNum

def exportList2CSV(csv_file,data_list,csv_columns=None):
	'''
	csv_file = filename of csv exported from list
	csv_column = column titles
	data_list = list data
	'''
	# export files
	# import csv

	with open(csv_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		if csv_columns != None:
			writer.writerow(csv_columns)
		for data in data_list:
			writer.writerow(data)   
	csvfile.close()
	return None

#################################################################################################################
### interpolation_lin_OK_UK
#################################################################################################################
def interpKrig(DEMname, interpType, xRange=None, yRange=None, zRange=None, gridMethodisN=True, Ngrid=50, Lgrid=1, stdMax=30, export=False, outputSingle=True, exportName=None, dp=3):
	'''
	process: lienar interpolation or kriging interpolation (ordinary and universal)

	Interpolation methods used: (for 1D and 2D)
	1. scipy - linear interpolation (short form = lin)
	2. pykrige - kriging ordinary   (short form = OK)
	3. pykrige - kriging universal  (short form = UK)

	Dimensions available:
	1. 2D - given: x 		calculate: y
	2. 3D - given: x, y 	calculate: z
	3. 4D - given: x, y, z	calculate: w  (kriging only)

	Mathematical model for semi-veriograms (kriging):
	1. linear						
	2. power			
	3. gaussian			
	4. spherical			
	5. exponential		
	6. circular (under-construction... no yet finished)		

	Inputs: 
		> DEMname = name of csv file
		> interpType = type of interpolation
		> xRange = range of x grid coordinates as list [min, max] (default = None)
		> yRange = range of y grid coordinates as list [min, max] (default = None)
		> zRange = range of z grid coordinates as list [min, max] (default = None)
		> gridMethodisN = method to create grid: number of grids, i.e. True, or length between grids, i.e. False (default = True)
		> Ngrid = number of grids between limits (default = 50)
		> Lgrid = length between grids (default = 1 unit of length)
		> stdMax = max acceptable error/standard deviation (default = 30)
		> cutVal = value to assign to interpolated locations exceeding stdMax (default = np.nan)
		> export option (default = True)
		> exportName = export csv file name (default = 'interpolated_'+DEMname)
		> dp = decimal point (default = 3)
	Output: 
		> interpolated contour (line or surface)
		> interpolated csv file

	How to write interpolType
	format: [dimension + space + short form interpolation type (+ space + semivariogram model)] in string format
	e.g.) '2 lin' = 2D linear interpolation 
		  '3 UK gaussian' = 3D ordinary kriging interpolation with linear semivariogram model
		  '4 OK linear' = 4D ordinary kriging interpolation with linear semivariogram model

	Note on xRange, yRange, zRange
		The range of values of x, y, and z for generating square grids can be pre-assigned 
		or used with the max and min values of each x, y, and z

	Note on grid generation
		The grids are squares. Therefore, the grids are generated in the following method:
			
			if Ngrid value is used
				1. find the min and max values of each given points range (x,y,z)
				2. find the spacing between grids for each orthogonal direction given each directions are divided with number N
				3. specified length = minimum value among spacing of each orthogonal direction
				4. specified steps = roundup((max value - min value) / specified length)

			elif Lgrid value is used
				1. find the min and max values of each given points range (x,y,z)
				2. specified length = user defined
				3. specified steps = roundup((max value - min value) / specified length)
			
			# grid generated in the following method
			1. subdivide the values of limits (max and min) of each orthogonal direction into specified steps
	'''

	''' set up '''
	# import modules
	# import numpy as np
	#from numpy import linspace,
	#import making_list_with_floats as makeList  # functions from making_list_with_floats.py 
	# from making_list_with_floats import csv2list, exportList2CSV

	# determine interpType
	# create a list of interpType info [dimensions, interpolation kind, semi-variagram model (OK and UK)]
	typeList = interpType.split(' ')		
	typeList[0] = int(typeList[0])
	if len(typeList) == 2 and typeList[1] in ['OK','UK']:
		typeList.append('linear')
	elif typeList[1] == 'lin':
		pass
	else:
		print('Error: check the interpType input')
		return None

	# sort information of given input file - needs to be in a matrix format
	#print(DEMname)
	'''
	if isinstance(DEMname, str): 
		inputFile = csv2list(DEMname)
	else:
		try:
			DEMname
		except NameError:
			inputFile = DEMname
	'''
	inputFile = DEMname
	inputCol = len(inputFile[0])

	# import csv file of given points and assign to each 
	if inputCol == 2:
		inputX, inputY = np.array(inputFile).T
		#inputZ = np.empty((len(inputX),))
		#inputW = np.empty((len(inputX),))
	elif inputCol == 3:
		inputX, inputY, inputZ = np.array(inputFile).T 
		if typeList[1] == 'lin': 
			inputXY = np.array(inputFile)[:,[0,1]]
	elif inputCol == 4:
		inputX, inputY, inputZ, inputW = np.array(inputFile).T
	else:
		print('Error: check your input file')
		return None

	# sort information depending on the dimension specified
	'''2D'''
	if typeList[0] == 2: 

		# check if user provided given range
		# for X
		if xRange == None:
			minX = min(inputX)
			maxX = max(inputX)
		else:
			minX = xRange[0]
			maxX = xRange[1]

		# number of grid points along each orthogonal direction
		if gridMethodisN == True:
			# find the minimum spacing
			spacing = abs((maxX-minX)/Ngrid)
			xStepN = round(abs(maxX-minX)/spacing)

		elif gridMethodisN == False:
			xStepN = round(abs(maxX-minX)/Lgrid)

		# coordinates of each grids in orthogonal directions
		gridXCoords = np.linspace(minX, maxX, xStepN, dtype=float)

		# interpolation process
		if typeList[1] == 'lin':
			# import relevent python modules for provided interpolation type and method chosen
			# from scipy.interpolate import interp1d

			# perform interpolation
			tempInterpolated = interp1d(inputX, inputY, bounds_error=False)
			interpolY = tempInterpolated(gridXCoords)
			stdY = None

		elif typeList[1] == 'OK':
			# import relevent python modules for provided interpolation type and method chosen
			# from pykrige.ok import OrdinaryKriging

			# perform interpolation
			tempInterpolated = OrdinaryKriging(inputX, np.zeros(inputX.shape), inputY, variogram_model=typeList[2])
			interpolY, stdY = tempInterpolated.execute('grid', gridXCoords, np.array([0.]))
			interpolY, stdY = interpolY[0], stdY[0]

		elif typeList[1] == 'UK':
			# import relevent python modules for provided interpolation type and method chosen
			# from pykrige.uk import UniversalKriging

			# perform interpolation
			tempInterpolated = UniversalKriging(inputX, np.zeros(inputX.shape), inputY, variogram_model=typeList[2])
			interpolY, stdY = tempInterpolated.execute('grid', gridXCoords, np.array([0.]))
			interpolY, stdY = interpolY[0], stdY[0]

		# for pykrige, eliminate points that has a large standard deviation
		if typeList[1] in ['OK', 'UK']:
			for loopYPred in range(len(interpolY)):
				if stdY[loopYPred] > stdMax:
					interpolY[loopYPred] = np.nan			

		# combine into single output file
		if typeList[1] in ['OK', 'UK']:
			outFile = (np.vstack((gridXCoords, interpolY, stdY)).T).tolist()
		else:
			outFile = (np.vstack((gridXCoords, interpolY, np.nan*np.ones((len(gridXCoords),)))).T).tolist()
		
		# export the interpolated data into csv file
		if export == True:
			if exportName == None:
				exportList2CSV('interpolated_'+interpType.replace(' ','_')+'_'+DEMname, outFile)
			else: 
				exportList2CSV(exportName+'.csv', outFile)

	'''3D'''
	if typeList[0] == 3:

		# check if user provided given range
		# for X
		if xRange == None:
			minX = min(inputX)
			maxX = max(inputX)
		else:
			minX = xRange[0]
			maxX = xRange[1]
		
		# for Y
		if yRange == None:
			minY = min(inputY)
			maxY = max(inputY)
		else:
			minY = yRange[0]
			maxY = yRange[1]

		# number of grid points along each orthogonal direction
		if gridMethodisN:
			# find the minimum spacing
			spacing = min(abs((maxX-minX)/Ngrid), abs((maxY-minY)/Ngrid))
			xStepN = round(abs((maxX-minX)/spacing))
			yStepN = round(abs((maxY-minY)/spacing))
		else:
			xStepN = round(abs((maxX-minX)/Lgrid))
			yStepN = round(abs((maxY-minY)/Lgrid))
			
		# coordinates of each grids in orthogonal directions
		gridXCoords = np.linspace(minX, maxX, xStepN, dtype=float)
		gridYCoords = np.linspace(minY, maxY, yStepN, dtype=float)

		# interpolation process
		if typeList[1] == 'lin':
			# import relevent python modules for provided interpolation type and method chosen
			# from scipy.interpolate import griddata

			meshgridX, meshgridY = np.meshgrid(gridXCoords, gridYCoords)

			# perform interpolation
			interpolZ = griddata(inputXY, inputZ, (meshgridX, meshgridY), method='linear')

			#tempInterpolated = interp2d(inputX, inputY, inputZ, bounds_error=False)
			#interpolZ = tempInterpolated(gridXCoords, gridYCoords)
			
			stdZ = None

		elif typeList[1] == 'OK':
			# import relevent python modules for provided interpolation type and method chosen
			# from pykrige.ok import OrdinaryKriging

			# perform interpolation
			tempInterpolated = OrdinaryKriging(inputX, inputY, inputZ, variogram_model=typeList[2])
			interpolZ, stdZ = tempInterpolated.execute('grid', gridXCoords, gridYCoords)

		elif typeList[1] == 'UK':
			# import relevent python modules for provided interpolation type and method chosen
			# from pykrige.uk import UniversalKriging

			# perform interpolation
			tempInterpolated = UniversalKriging(inputX, inputY, inputZ, variogram_model=typeList[2])
			interpolZ, stdZ = tempInterpolated.execute('grid', gridXCoords, gridYCoords)

		# for pykrige, eliminate points that has a large standard deviation
		if typeList[1] in ['OK', 'UK']:
			for loopZPred1 in range(len(interpolZ)):
				for loopZPred2 in range(len(interpolZ[0])):
					if stdZ[loopZPred1][loopZPred2] > stdMax:
						interpolZ[loopZPred1][loopZPred2] = np.nan		

		# combine all coordinates into single output file format
		outFile = []
		for loop31 in range(len(interpolZ)):
			for loop32 in range(len(interpolZ[0])):
				# combine into single output file
				if typeList[1] in ['OK', 'UK']:
					outFile.append([gridXCoords[loop32], gridYCoords[loop31], interpolZ[loop31][loop32], stdZ[loop31][loop32]])
				else:
					outFile.append([gridXCoords[loop32], gridYCoords[loop31], interpolZ[loop31][loop32], np.nan])					
		
		# export the interpolated data into csv file
		if export == True:
			if exportName == None:
				exportList2CSV('interpolated_'+interpType.replace(' ','_')+'_'+DEMname, outFile)
			else: 
				exportList2CSV(exportName+'.csv', outFile)
		
	'''4D'''
	if typeList[0] == 4:

		# check if user provided given range
		# for X
		if xRange == None:
			minX = min(inputX)
			maxX = max(inputX)
		else:
			minX = xRange[0]
			maxX = xRange[1]
		
		# for Y
		if yRange == None:
			minY = min(inputY)
			maxY = max(inputY)
		else:
			minY = yRange[0]
			maxY = yRange[1]
		
		# for Z
		if zRange == None:
			minZ = min(inputZ)
			maxZ = max(inputZ)
		else:
			minZ = zRange[0]
			maxZ = zRange[1]

		# number of grid points along each orthogonal direction
		if gridMethodisN == True:
			# find the minimum spacing
			spacing = min([abs((maxX-minX)/Ngrid), abs((maxY-minY)/Ngrid), abs((maxZ-minZ)/Ngrid)])
			xStepN = round(abs((maxX-minX)/spacing))
			yStepN = round(abs((maxY-minY)/spacing))
			zStepN = round(abs((maxZ-minZ)/spacing))
		elif gridMethodisN == False:
			xStepN = round(abs((maxX-minX)/Lgrid))
			yStepN = round(abs((maxY-minY)/Lgrid))
			zStepN = round(abs((maxZ-minZ)/Lgrid))

		# coordinates of each grids in orthogonal directions
		gridXCoords = np.linspace(minX, maxX, xStepN, dtype=float)
		gridYCoords = np.linspace(minY, maxY, yStepN, dtype=float)
		gridZCoords = np.linspace(minZ, maxZ, zStepN, dtype=float)	

		# interpolation process
		if typeList[1] == 'lin':
			print('sorry - not completed yet')
			return None

		elif typeList[1] == 'OK':
			# import relevent python modules for provided interpolation type and method chosen
			# from pykrige.ok3d import OrdinaryKriging3D

			# perform interpolation
			tempInterpolated = OrdinaryKriging3D(inputX, inputY, inputZ, inputW, variogram_model=typeList[2])
			interpolW, stdW = tempInterpolated.execute('grid', gridXCoords, gridYCoords, gridZCoords)

		elif typeList[1] == 'UK':
			# import relevent python modules for provided interpolation type and method chosen
			# from pykrige.uk3d import UniversalKriging3D

			# perform interpolation
			tempInterpolated = UniversalKriging3D(inputX, inputY, inputZ, inputW, variogram_model=typeList[2])
			interpolW, stdW = tempInterpolated.execute('grid', gridXCoords, gridYCoords, gridZCoords)

		# for pykrige, eliminate points that has a large standard deviation
		if typeList[1] in ['OK', 'UK']:
			for loopWPred1 in range(len(interpolW)):
				for loopWPred2 in range(len(interpolW[0])):
					for loopWPred3 in range(len(interpolW[0][0])):
						if stdW[loopWPred1][loopWPred2][loopWPred3] > stdMax:
							interpolW[loopWPred1][loopWPred2][loopWPred3] = np.nan		

		# combine all coordinates into single output file format
		outFile = []
		for loop31 in range(len(interpolW)):
			for loop32 in range(len(interpolW[0])):
				for loop33 in range(len(interpolW[0][0])):
					# combine into single output file
					if typeList[1] in ['OK', 'UK']:
						outFile.append([gridXCoords[loop33], gridYCoords[loop32], gridZCoords[loop31], interpolW[loop31][loop32][loop33], stdW[loop31][loop32][loop33]])
					else:
						outFile.append([gridXCoords[loop33], gridYCoords[loop32], gridZCoords[loop31], interpolW[loop31][loop32][loop33], np.nan])						
		
		# export the interpolated data into csv file
		if export == True:
			if exportName == None:
				exportList2CSV('interpolated_'+interpType.replace(' ','_')+'_'+DEMname, outFile)
			else: 
				exportList2CSV(exportName+'.csv', outFile)

	
	'''output files'''
	if outputSingle == False:
		if typeList[0] == 2:
			return outFile, gridXCoords, interpolY, stdY
		elif typeList[0] == 3:
			return outFile, gridXCoords, gridYCoords, interpolZ, stdZ
		elif typeList[0] == 4:
			return outFile, gridXCoords, gridYCoords, gridZCoords, interpolW, stdW
	else:
		return outFile

#################################################################################################################
### geoFileConvert_20201028
#################################################################################################################
###########################################################################
## conversion from xyz (csv) file format into mesh grid
###########################################################################
# convert xyz to mesh grid matrix
def xyz2mesh(inFileName, exportAll=False, dtype_opt=float):
	'''
	# input
		inFileName		:	input csv file name (.csv file) already ready to generate as mesh
		exportAll		:	export all necessary infos (default = False)
	# output
		gridUniqueX 	:	unqiue list of x coordinates in the mesh
		gridUniqueY 	:	unqiue list of y coordinates in the mesh
		outputZ			:	output grid array (rowN=y, colN=x, element=z)
	'''
	# import python modules
	# import numpy as np
	# import csv
	#from making_list_with_floats import csv2list

	# check inFileName is a variable or a csv file to be imported
	if isinstance(inFileName, str):  
		try: 
			dataset = np.array(csv2list(inFileName))
		except:
			dataset = np.array(csv2list(inFileName+'.csv'))
	else:
		try:
			inFileName	
		except NameError:
			pass
		else:
			if isinstance(inFileName, list):
				dataset = np.array(inFileName)
			elif isinstance(inFileName, np.ndarray):
				dataset = inFileName

	# create a unique list of x and y coordinates
	gridUniqueX = np.unique(dataset[:,0])
	gridUniqueY = np.unique(dataset[:,1])
	
	# place each unique grid into dictionary for easy search
	gridUniqueX_dict = {}
	for idx,loopx in enumerate(gridUniqueX):
		gridUniqueX_dict[loopx] = idx

	gridUniqueY_dict = {}
	for idy,loopy in enumerate(gridUniqueY):
		gridUniqueY_dict[loopy] = idy

	# go through each line of csv file and place them into a grid format
	# row number = y-coordinates
	# col number = x-coordinates
	outputZ = np.zeros((len(gridUniqueY),len(gridUniqueX)),dtype=dtype_opt)
	for xi,yi,zi in zip(dataset[:,0],dataset[:,1],dataset[:,2]):
		outputZ[gridUniqueY_dict[yi]][gridUniqueX_dict[xi]] = zi

	if exportAll:
		deltaX = abs(gridUniqueX[0]-gridUniqueX[1])		# spacing between x grids
		deltaY = abs(gridUniqueY[0]-gridUniqueY[1])		# spacing between y grids
		return outputZ, gridUniqueX, gridUniqueY, deltaX, deltaY
	else:
		return outputZ

###########################################################################
## conversion between xyz/csv and lidar (las) file
###########################################################################
# convert las to xyz
def las2xyz(inFileName, outFileName=None, outFileFormat='csv', saveOutputFile=False):

	'''
	# input
		inFileName			:	input las file name (.las file)
		outFileName			:	output xyz file name 
		outFileFormat		:	output xyz file format (default = 'csv')
									if 'csv' -> comma delimited csv file (.csv)
									if 'txt' -> tab delimited txt file (.txt)
		saveOutputFile		:	save the convert file (default = False)
	# output
		outFileName			:	output xyz file name 
	'''
	
	# import python modules
	# import laspy
	# import numpy as np

	# convert las to csv i.e. xyz
	inFile = laspy.file.File(inFileName+'.las', mode="r")
	# inFile = laspy.file.File(inFileName, mode="r")

	# inFileX = np.array((inFile.X - inFile.header.offset[0])*inFile.header.scale[0])
	# inFileY = np.array((inFile.Y - inFile.header.offset[1])*inFile.header.scale[1])
	# inFileZ = np.array((inFile.Z - inFile.header.offset[2])*inFile.header.scale[2])

	inFileX = np.array(inFile.x)
	inFileY = np.array(inFile.y)
	inFileZ = np.array(inFile.z)

	dataset = np.vstack([inFileX, inFileY, inFileZ]).transpose()
	#print(dataset[:10])

	if saveOutputFile:
		# import csv

		if outFileName == None:
			outFileNameF = inFileName+'_XYZ'
		else:
			outFileNameF = outFileName

		if outFileFormat == 'csv':
			with open(outFileNameF+'.csv', 'w', newline='') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')
				for data in dataset:
					writer.writerow(data) 
		elif outFileFormat == 'txt':
			with open(outFileNameF+'.txt', 'w', newline='') as csvfile:
				writer = csv.writer(csvfile, delimiter='\t')
				for data in dataset:
					writer.writerow(data) 

	inFile.close()

	return dataset

def nonzero_int(value):
	# import decimal 

	if abs(decimal.Decimal(value).as_tuple().exponent):
		return 0

	intVal_list = list(str(int(value)))
	intVal_list.reverse()

	countZero = 0
	for idx in range(len(intVal_list)):
		if int(intVal_list[idx]) == 0:
			countZero += 1
		elif int(intVal_list[idx]) != 0:
			break

	return countZero

# convert csv to las
def xyz2las(inFileName, inFileFormat='csv', outFileName=None, offsets=[None,None,None], scales=[None,None,None], maxDP=3):

	'''
	# input
		inFileName			:	input csv/xyz file name 

			Note: there is bug in the laspy module
				if there are more than 5 digits on the left side of the decimal point (i.e. value >= million),
				then there is error occuring - currently fixed by changing the scaling factor

		inFileFormat 		:	type of file input (default = 'csv')
									if 'csv' = input csv file (comma delimiter)
									if 'txt' = input txt file (tab delimiter)

		outFileName			:	output xyz file name 
	
		Note: actual_value = (las_value - offset_value)*scale_value

		offset				:	offset value [xmin,ymin,zmin]	(default = [None,None,None])
									if None -> no offset; therefore, offset = 0
									if 'min' -> value minimum as offset 
									if userdefined [offset in x, offset in y, offset in z] -> just assign offset user requests

								example if offset=[None,'min',0.01], then offset = [0 (no offset), minimum Y value, 0.01 (user specified)]
						
		scale				:	precision scale factor	(default = [None,None,None])
									if None -> compute the lowest decimal points of inputs 
									if userdefined [scale in x, scale in y, scale in z] -> just assign precision user requests 

								example if offset=[None,0.01,0.01], then offset = [based on X value with lowest decimal, 0.01 (user specified), 0.01 (user specified)]

		maxDP				:	maximum decimal place for automatically computing the precision scale factor: int number (default = 3)

	# output
		outFileName			:	output xyz file name 
	'''
	
	# import python modules
	# import laspy
	#import csv
	#from making_list_with_floats import csv2list, txt2list
	# import numpy as np

	# check inFileName is a variable or a csv file to be imported
	'''
	try:
		inFileName
	except NameError:
		dataset = np.array(csv.reader(inFileName+'.csv'))
	else:
		if isinstance(inFileName, list):
			dataset = np.array(inFileName)
		elif isinstance(inFileName, np.ndarray):
			dataset = inFileName
	'''
	# import csv or txt file
	if inFileFormat == 'csv':
		dataset = np.array(csv2list(inFileName+'.csv'))
	elif inFileFormat == 'txt':
		dataset = np.array(txt2list(inFileName+'.txt'))

	# create blank las file to be filled 
	hdr = laspy.header.Header()		# generate las file header

	# assign name to the output las file
	if outFileName == None:
		outFileNameF = 'output_xyz2las'
	else:
		outFileNameF = outFileName

	# create output las file to be written
	outfile = laspy.file.File(outFileNameF+'.las', mode="w", header=hdr)

	# sort each XYZ values into individual numpy 1-D array
	allx = dataset.transpose()[0]			
	ally = dataset.transpose()[1]
	allz = dataset.transpose()[2]

	## offset 

	# compute minimum XYZ values if there is 'min' option in the offset input
	if 'min' in offsets:
		xmin = np.floor(allx.min())
		ymin = np.floor(ally.min())
		zmin = np.floor(allz.min())
		minXYZ = [xmin, ymin, zmin] 

	offsetList_f = [0,0,0]
	for loopOffset in range(3):

		if offsets[loopOffset] == None:	# no offset
			offsetList_f[loopOffset] = 0

		elif offsets[loopOffset] == 'min':	# use minimum value as offset
			offsetList_f[loopOffset] = minXYZ[loopOffset] 

		else:
			offsetList_f[loopOffset] = offsets[loopOffset]

	outfile.header.offset = offsetList_f

	## scale

	# compute max decimal point of XYZ values if there is None option in the scale input
	sampleSize = 20
	if None in scales:

		# take a random value along the XYZ datapoints
		randomX = allx[np.random.randint(0, high=len(allx), size=sampleSize)]
		randomY = ally[np.random.randint(0, high=len(ally), size=sampleSize)]
		randomZ = allz[np.random.randint(0, high=len(allz), size=sampleSize)]
		randomXYZ = [randomX, randomY, randomZ]

		# find the decimal place of each random XYZ value, but set maximum decimal point as 3 
		# import decimal 
		# from scipy.stats import mode

		dpN = [[],[],[]]
		for loopDP in range(sampleSize):	
			dpN[0].append(abs(decimal.Decimal(randomX[loopDP]).as_tuple().exponent))
			dpN[1].append(abs(decimal.Decimal(randomY[loopDP]).as_tuple().exponent))
			dpN[2].append(abs(decimal.Decimal(randomZ[loopDP]).as_tuple().exponent))
		
		#dpList = [min(mode(dpN[0],axis=None)[0][0] - 2, maxDP), min(mode(dpN[1],axis=None)[0][0] - 2, maxDP), min(mode(dpN[2],axis=None)[0][0] - 2, maxDP)]
		dpList = [min(int(np.floor(0.5*(max(dpN[0])+min(dpN[0])))), maxDP),
					min(int(np.floor(0.5*(max(dpN[1])+min(dpN[1])))), maxDP), 
					min(int(np.floor(0.5*(max(dpN[2])+min(dpN[2])))), maxDP)]
		#print(dpList)

		# find the number of interger points on the left side of the decimal point
		ipN = [[],[],[]]
		for loopIP in range(sampleSize):	
			ipN[0].append(len(str(int(randomX[loopIP]))))
			ipN[1].append(len(str(int(randomY[loopIP]))))
			ipN[2].append(len(str(int(randomZ[loopIP]))))
		ipList = [mode(ipN[0],axis=None)[0][0], mode(ipN[1],axis=None)[0][0], mode(ipN[2],axis=None)[0][0]]

		# find the number of significant points data (nonzero value) in the integer value
		sfN = [[],[],[]]
		for loopSF in range(sampleSize):	
			sfN[0].append(nonzero_int(randomX[loopSF]))
			sfN[1].append(nonzero_int(randomY[loopSF]))
			sfN[2].append(nonzero_int(randomZ[loopSF]))
		sfList = [min(sfN[0]), min(sfN[1]), min(sfN[2])]

		# find maximum decimal place from XYZ value and place as scale 
		scale_None = round(0.1**max(dpList), max(dpList))
	
	scaleList_f = [0,0,0]
	for loopScale in range(3):

		if scales[loopScale] == None:	
			
			if dpList[loopScale] == 0:	# if there is no decimal point
				scaleList_f[loopScale] = 10**sfList[loopScale]
			
			else:	# if there is decimla point
				scaleList_f[loopScale] = round(0.1**dpList[loopScale], dpList[loopScale])
		
		else:	# user defined scale
			scaleList_f[loopScale] = scales[loopScale]
			
	outfile.header.scale = scaleList_f

	# input 
	outfile.x = allx
	outfile.y = ally
	outfile.z = allz

	outfile.close()

	return None

###########################################################################
## conversion between xyz to asc (ESRI ASCII grid) file format 
###########################################################################
# convert asc to xyz
def asc2xyz(inFileName, outFileName=None, saveOutputFile=False):
	'''
	# input
		inFileName			:	input ESRI ascii file name (.asc file)
		outFileName			:	output xyz file name (default = None)
		saveOutputFile		:	save the convert file (default = False)
	# output
		outFileName			:	output xyz file name
	'''

	# import python modules
	# import numpy as np

	with open(inFileName+'.asc', 'r') as myfile:
		data=myfile.read().replace('\n', ',').split(',')

	# sort list of files
	ncols = int(data[0].split(' ')[1])		# for y axis
	nrows = int(data[1].split(' ')[1])		# for x axis
	xllcorner = float(data[2].split(' ')[1])
	yllcorner = float(data[3].split(' ')[1])
	cellsize = float(data[4].split(' ')[1])
	nodata_value = float(data[5].split(' ')[1])

	tempZgrid = []
	for zRow in data[6:len(data)]:
		tempZrow = zRow.split(' ')
		tempZgrid.append([ float(x) for x in tempZrow ])
	zMesh = np.array(tempZgrid)

	# create grid
	xGrids = np.linspace(xllcorner, xllcorner+cellsize*nrows, nrows)
	yGrids = np.linspace(yllcorner, yllcorner+cellsize*ncols, ncols)

	# file
	outfile = []
	for i in range(len(xGrids)):
		for j in range(len(yGrids)):
			outfile.append([xGrids[i], yGrids[j], zMesh[j][i]])

	if saveOutputFile:
		# import csv

		if outFileName == None:
			outFileNameF = inFileName+'_XYZ'
		else:
			outFileNameF = outFileName

		with open(outFileNameF+'.csv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for data1 in outfile:
				writer.writerow(data1) 

	return outfile

# convert xyz to asc and export asc file
def xyz2asc(inFileName, outFileName=None, interpType='3 lin', cellSize=1.0, user_nodata_value=-9999):

	'''
	# input
		inFileName			:	input xyz file name
		outFileName			:	output ESRI ascii file name 
		interpType			:	type of interpolation to perform to create grid (default = '3 lin')
		cellSize			:	size of cells (default = 1.0)
	# output
		outFileName			:	output asc file name
	'''

	# import python modules
	# import numpy as np
	# from interpolation_lin_OK_UK import interpKrig

	# import csv/xyz file and interpolate to create a grid 
	outFile, gridXCoords, gridYCoords, interpolZ, stdZ = interpKrig(inFileName, interpType, gridMethodisN=False, Lgrid=cellSize, stdMax=10000, outputSingle=False) 

	# create header 
	header = "ncols %s\n" % len(gridYCoords)	# number of grids along y axis
	header += "nrows %s\n" % len(gridXCoords)	# number of grids along x axis
	header += "xllcorner %s\n" % min(gridXCoords)	# X-coordinate of the origin (by center or lower left corner of the cell)
	header += "yllcorner %s\n" % min(gridYCoords)	# Y-coordinate of the origin (by center or lower left corner of the cell)
	header += "cellsize %s\n" % cellSize 			# distance between each grids
	header += "nodata_value %s" % user_nodata_value

	if outFileName == None:
		outFileNameF = 'output_xyz2'+'enri_ascii_'+interpType
	else:
		outFileNameF = outFileName

	np.savetxt(outFileNameF+'.asc', interpolZ, header=header, fmt="%1.2f", comments='')

	return None

###########################################################################
## conversion between xyz to grd (Surfer grid) file format 
###########################################################################
# convert grd to xyz
def grd2xyz(inFileName, headDataOutput=False, outFileName=None, saveOutputFile=False):
	'''
	# input
		inFileName			:	input surfer grd file name (.grd file)
		headDataOutput			:	extract grd gead data (default = False)
		outFileName			:	output xyz file name 
		saveOutputFile		:	save the convert file (default = False)
	# output
		outFileName			:	output xyz file name
	'''

	# import python modules
	# import numpy as np

	# if isinstance(inFileName, str): 
	# 	inFileNameFormat = inFileName.split('.')

	# 	if len(inFileNameFormat) == 1:
	# 		with open(inFileName+'.grd', 'r') as myfile:
	# 			data=myfile.read().split('\n')
	# 	else:
	# 		with open(inFileName, 'r') as myfile:
	# 			data=myfile.read().split('\n')

	with open(inFileName+'.grd', 'r') as myfile:
		data=myfile.read().split('\n')

	#print(data)
	# sort list of files

	## step 1: find the deliminator of the grd file
	allItems = list(data[1])

	if (' ' in allItems) and ('\t' not in allItems):
		delimGrd = ' '
	elif (' ' not in allItems) and ('\t' in allItems):
		delimGrd = '\t'
	else:
		print('something wrong with grd file on tab and space deliminator')
		assert(1!=1)

	Nx = int(data[1].split(delimGrd)[0])
	Ny = int(data[1].split(delimGrd)[1])
	xMin = float(data[2].split(delimGrd)[0])
	xMax = float(data[2].split(delimGrd)[1])
	yMin = float(data[3].split(delimGrd)[0])
	yMax = float(data[3].split(delimGrd)[1])
	zMin = float(data[4].split(delimGrd)[0])
	zMax = float(data[4].split(delimGrd)[1])

	#print(inFileName)

	# create grid for z axis (elevation)
	if delimGrd == ' ':
		# find all indices where there should change in row
		indices = [i for i, x in enumerate(data) if x == '']  
		#print(len(indices))
		tempZgrid = []

		for loopN in range(len(indices)):
			#print(indices[loopN])
			tempZrow = ''
			# starting iteration list index
			if loopN == 0:
				startID = 5
			else:
				startID = indices[loopN-1]+1

			# join all the string
			for loopNN in range(startID, indices[loopN]):
				tempZrow += str(data[loopNN])

			tempZrowList = (tempZrow.split(' '))
			tempZrowList.pop()
			tempZgrid.append([ float(x) for x in tempZrowList ])

		tempZgrid.pop()

	elif delimGrd == '\t':
		tempZgrid = []

		for loopN in range(5,len(data)):
			rowData = data[loopN].split(delimGrd)
			rowData.pop()

			tempZgrid.append([ float(x) for x in rowData ])

	zMesh = np.array(tempZgrid)
	#print(tempZgrid)

	# create grid
	xGrids = np.linspace(xMin, xMax, Nx)
	yGrids = np.linspace(yMin, yMax, Ny)

	# file
	outfile = []
	'''
	for i in range(len(xGrids)):
		for j in range(len(yGrids)):
			outfile.append([xGrids[i], yGrids[j], zMesh[j][i]])
	'''
	#print('grd2xyz  len(yGrids),len(xGrids),len(zMesh),len(zMesh[0])')
	#print(len(yGrids),len(xGrids),len(zMesh),len(zMesh[0]))
	#for j in range(len(yGrids)-1,-1,-1):
	for j in range(len(yGrids)):
		for i in range(len(xGrids)):
			outfile.append([xGrids[i], yGrids[j], zMesh[j][i]])

	# output csv file
	if saveOutputFile:
		# import csv

		if outFileName == None:
			outFileNameF = inFileName+'_XYZ'
		else:
			outFileNameF = outFileName

		with open(outFileNameF+'.csv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for data1 in outfile:
				writer.writerow(data1) 
	
	if headDataOutput:
		return outfile, [Nx, Ny, xMin, xMax, yMin, yMax, zMin, zMax]
	else:
		return outfile

# convert grd to numpy array and find Zcoordinate value from given XY coordinates 
def grd2pointXY(inFileName, XYcoord, interpType='lin', data_diff=2, single_output=True):
	'''
	# input
		inFileName		:	input surfer grd file name (.grd file)
		
		XYcoord 		:	list of x and y coordinates (in list form) -> [X, Y]
			e.g. XYcoord = [[0,1],[1,2]]   find all z value at points [0,1] and [1,2]
		
		interpType		:	3D interpolation method to use if not exact (default = 'lin')
			'lin' -> scipy.interpolate.griddata function with linear method
			'cubic' -> scipy.interpolate.griddata function with cubic method
			'OK xxx'  -> Ordinary Kriging with semi-variance model of xxx (pykrige)
				e.g. 'OK linear' -> Ordinate Kriging with linear semi-variance model
			'UK xxx'  -> Universal Kriging with semi-variance model of xxx (pykrige)
				e.g. 'UK gaussian' -> Universal Kriging with gaussian semi-variance model
	
			Mathematical model for semi-veriograms (kriging):
			1. linear						
			2. power			
			3. gaussian			
			4. spherical			
			5. exponential	

		data_diff		:	interval at which the sample is selected from the whole data 
			e.g. data_diff = 2 -> points at index above and below 2 of the selected XY point will be taken as consideration for interpolations

	# output
		
		output 			:	list of XYZ output value from grd file (exact or interpolated) [X, Y, Z]
			e.g. output = [[0,1,0],[1,2,1]]
		dim_value		:	list of Z output value from grd file (exact or interpolated) [Z]
	'''

	# import python modules
	# import numpy as np

	if isinstance(inFileName, str): 
		inFileNameFormat = inFileName.split('.')

		if len(inFileNameFormat) == 1:
			with open(inFileName+'.grd', 'r') as myfile:
				data=myfile.read().split('\n')
		else:
			with open(inFileName, 'r') as myfile:
				data=myfile.read().split('\n')

	#print(data)
	# sort list of files

	## step 1: find the deliminator of the grd file
	allItems = list(data[1])

	if (' ' in allItems) and ('\t' not in allItems):
		delimGrd = ' '
	elif (' ' not in allItems) and ('\t' in allItems):
		delimGrd = '\t'
	else:
		print('something wrong with grd file on tab and space deliminator')
		assert(1!=1)

	# sort grid information
	Nx = int(data[1].split(delimGrd)[0])
	Ny = int(data[1].split(delimGrd)[1])
	xMin = float(data[2].split(delimGrd)[0])
	xMax = float(data[2].split(delimGrd)[1])
	yMin = float(data[3].split(delimGrd)[0])
	yMax = float(data[3].split(delimGrd)[1])
	# zMin = float(data[4].split(delimGrd)[0])
	# zMax = float(data[4].split(delimGrd)[1])

	#print(inFileName)

	# create grid for z axis (elevation)
	if delimGrd == ' ':
		# find all indices where there should change in row
		indices = [i for i, x in enumerate(data) if x == '']  
		#print(len(indices))
		tempZgrid = []

		for loopN in range(len(indices)):
			#print(indices[loopN])
			tempZrow = ''
			# starting iteration list index
			if loopN == 0:
				startID = 5
			else:
				startID = indices[loopN-1]+1

			# join all the string
			for loopNN in range(startID, indices[loopN]):
				tempZrow += str(data[loopNN])

			tempZrowList = (tempZrow.split(' '))
			tempZrowList.pop()
			tempZgrid.append([ float(x) for x in tempZrowList ])

		tempZgrid.pop()

	elif delimGrd == '\t':
		tempZgrid = []

		for loopN in range(5,len(data)):
			rowData = data[loopN].split(delimGrd)
			rowData.pop()

			tempZgrid.append([ float(x) for x in rowData ])

	zMesh = np.array(tempZgrid)
	# row = y coordinates, col = x coordinates
	dim_value = []

	## loop through given XY corodinates to find Z value
	for loopXY in range(len(XYcoord)):	

		# compute index location of XY coordinates in the mesh
		# compute the step of interval along the x and y grids
		xIDX = (XYcoord[loopXY][0] - xMin)/((xMax-xMin)/(Nx-1))
		yIDX = (XYcoord[loopXY][1] - yMin)/((yMax-yMin)/(Ny-1))

		# check whether xIDX or yIDX is integer
		checkX = (abs(xIDX-int(xIDX)) == 0)		# check xIDX is int
		checkY = (abs(yIDX-int(yIDX)) == 0)		# check yIDX is int

		# if both are integer, take the z value directly from zMesh
		if checkX and checkY:
			XYcoord[loopXY].append(zMesh[int(yIDX)][int(xIDX)])
			dim_value.append(zMesh[int(yIDX)][int(xIDX)])

		# else if not use interpolation
		else:
			# x and y grid coordinates
			gridXCoords = np.linspace(xMin, xMax, Nx, dtype=float)
			gridYCoords = np.linspace(yMin, yMax, Ny, dtype=float)

			# compute index for 
			if checkX:
				section_gridXCoords_idx = np.linspace((np.floor(xIDX)-data_diff),(np.ceil(xIDX)+data_diff),(2*data_diff+1),dtype=int)
			else:
				section_gridXCoords_idx = np.linspace((np.floor(xIDX)-data_diff),(np.ceil(xIDX)+data_diff),(2*data_diff+2),dtype=int)

			if checkY:
				section_gridYCoords_idx = np.linspace((np.floor(yIDX)-data_diff),(np.ceil(yIDX)+data_diff),(2*data_diff+1),dtype=int)
			else:
				section_gridYCoords_idx = np.linspace((np.floor(yIDX)-data_diff),(np.ceil(yIDX)+data_diff),(2*data_diff+2),dtype=int)

			#print(section_gridXCoords_idx)
			#print(section_gridYCoords_idx)

			# determine the interpolation method
			# create a list of interpType info [interpolation kind, semi-variagram model (OK and UK)]
			typeList = interpType.split(' ')		
			if len(typeList) == 1 and typeList[0] in ['OK','UK']:
				typeList.append('linear')

			## create input file
			# XYZ format	
			inputFile = []		
			for j in section_gridYCoords_idx:
				for i in section_gridXCoords_idx:
					inputFile.append([gridXCoords[i], gridYCoords[j], zMesh[j][i]])

			#print(inputFile)

			# column of inptues in XYZ coordiantes
			inputX, inputY, inputZ = np.array(inputFile).T 
			#print(inputX, inputY, inputZ)

			# for 'lin' or 'cubic' method combine X and Y column
			if typeList[0] in ['lin','cubic']: 
				inputXY = np.array(inputFile)[:,[0,1]]

			# interpolation process
			if typeList[0] == 'lin':

				# import relevent python modules for provided interpolation type and method chosen
				# from scipy.interpolate import griddata

				# perform interpolation
				interpolZ = griddata(inputXY, inputZ, (XYcoord[loopXY][0], XYcoord[loopXY][1]), method='linear')

				# add Z value 
				interpolZ = interpolZ.tolist()
				#XYcoord[loopXY].append(interpolZ)
				dim_value.append(interpolZ)

			
			elif typeList[0] == 'cubic':
				# import relevent python modules for provided interpolation type and method chosen
				# from scipy.interpolate import griddata

				# perform interpolation
				interpolZ = griddata(inputXY, inputZ, (XYcoord[loopXY][0], XYcoord[loopXY][1]), method='cubic')
				
				# add Z value 
				interpolZ = interpolZ.tolist()
				XYcoord[loopXY].append(interpolZ)
				dim_value.append(interpolZ)

			elif typeList[0] == 'OK':
				# import relevent python modules for provided interpolation type and method chosen
				# from pykrige.ok import OrdinaryKriging

				# perform interpolation
				try:
					tempInterpolated = OrdinaryKriging(inputX, inputY, inputZ, variogram_model=typeList[1])
					#interpolZ, stdZ = tempInterpolated.execute('grid', section_gridXCoords, section_gridYCoords)
					interpolZ, stdZ = tempInterpolated.execute('points', XYcoord[loopXY][0], XYcoord[loopXY][1])
				except:
					if abs(inputZ.max() - inputZ.min()) < 0.0001:
						interpolZ=np.array([0.0])
					else:
						interpolZ=np.array([None])
		
				# add Z value 
				interpolZ = interpolZ.tolist()
				XYcoord[loopXY].append(interpolZ[0])
				dim_value.append(interpolZ[0])

			elif typeList[0] == 'UK':
				# import relevent python modules for provided interpolation type and method chosen
				# from pykrige.uk import UniversalKriging

				# perform interpolation
				try:
					tempInterpolated = UniversalKriging(inputX, inputY, inputZ, variogram_model=typeList[1])
					interpolZ, stdZ = tempInterpolated.execute('points', XYcoord[loopXY][0], XYcoord[loopXY][1])
				except:
					if abs(inputZ.max() - inputZ.min()) < 0.0001:
						interpolZ=np.array([0.0])
					else:
						interpolZ=np.array([None])

				# add Z value 
				interpolZ = interpolZ.tolist()
				XYcoord[loopXY].append(interpolZ[0])
				dim_value.append(interpolZ[0])
	
	# output
	if single_output == True:
		return dim_value
	else:
		return XYcoord, dim_value

# convert xyz to grd and export grd file
def xyz2grd(inFileName, offset=None, outFileName=None, interp=True, interpType='3 lin', userNx=None, userNy=None):
	'''
	# input
		inFileName			:	input xyz file name
		offset				:	offset of x and y coordinates to make origin (i.e. leftmost x and bottom y) into (0,0) coordinates (default=None)
			for user requires to input of offset:	offset=(distance to move x, distance to move y)
			if offset is None, then the computer will automatically assest whether the origin is at (0,0) and automatically offset the x and y coordinates 
		outFileName			:	output surfer grd file name 
		interp 				:	data requires interpolation. If True, then interpolation is conducted (default=True)
		interpType			:	type of interpolation to perform to create grid (default = '3 lin')
		userNx				:	number of spacing in x axis (default = None)
		userNy				:	number of spacing in y axis (default = None)
	
	if userNx or userNy is assigned, the number of spacing is equivalent to the unit distance between the max and min value
	for example,	xMin=0,  xMax=100   ->  userNx = round(xMax-xMin)
		
	# output
		outFileName			:	output asc file name
	'''

	# import python modules
	# import numpy as np
	# from interpolation_lin_OK_UK import interpKrig
	#from making_list_with_floats import csv2list

	# check inFileName is a variable or a csv file to be imported
	dataset = np.array(csv2list(inFileName))
	'''
	try:
		inFileName
	except NameError:
		dataset = np.array(csv.reader(inFileName+'.csv'))
	else:
		if isinstance(inFileName, list):
			dataset = np.array(inFileName)
		elif isinstance(inFileName, np.ndarray):
			dataset = inFileName
	'''
	#print(dataset.transpose())

	allx = dataset.transpose()[0]
	ally = dataset.transpose()[1]
	allz = dataset.transpose()[2]

	# calculate offset value
	if offset == None:
		offsetX = allx.min()
		offsetY = ally.min()
	else:
		offsetX = offset[0]
		offsetY = offset[1]

	if interp:
		# number of spacings in x and y direction
		if userNx == None:
			Nx = abs(round(allx.max()) - round(allx.min()))+1
			Ny = abs(round(ally.max()) - round(ally.min()))+1
		elif userNx != None and isinstance(userNx, int) and isinstance(userNy, int):
			Nx = userNx
			Ny = userNy

		# find minimum spacing distance
		userLgrid = max([abs((allx.max() - allx.min())/(Nx)), abs((ally.max() - ally.min())/(Ny))])
		#min([abs((allx.max() - allx.min())/(Nx)), abs((ally.max() - ally.min())/(Ny))])

		#print(abs((allx.max() - allx.min())/(Nx)), abs((ally.max() - ally.min())/(Ny)), Nx, Ny, (userLgrid))
		#print(abs((allx.max() - allx.min())/(0.9987819732034104)), abs((ally.max() - ally.min())/(0.9987819732034104)))
		#print(abs((allx.max() - allx.min())/(0.99812382739212)), abs((ally.max() - ally.min())/(0.99812382739212)))

		# import csv/xyz file and interpolate to create a grid 
		#outFile, gridXCoords, gridYCoords, interpolZ, stdZ = interpKrig(dataset, interpType, gridMethodisN=True, Ngrid=(min([Nx,Ny])), stdMax=10000, outputSingle=False)
		outFile, gridXCoords, gridYCoords, interpolZ, stdZ = interpKrig(dataset, interpType, gridMethodisN=False, Lgrid=userLgrid, stdMax=10000, outputSingle=False)
		

		#print('xyz2grd   len(gridXCoords), len(gridYCoords), len(interpolZ), len(interpolZ[0])')
		#print(len(gridXCoords), len(gridYCoords), len(interpolZ), len(interpolZ[0]))
		#print((allz.min(), allz.max()))
		#print((interpolZ.min(), interpolZ.max()))
		#print(np.where(interpolZ == 1.8))
		#print(interpolZ[45][445:448])
	else:
		interpolZ, gridXCoords, gridYCoords, deltaX, deltaY = xyz2mesh(dataset, exportAll=True)
	#print(interpolZ.tolist())

	# create header 
	header = 'DSAA\n'
	header += "%s %s\n" % (len(gridXCoords), len(gridYCoords))			# Nx and Ny
	header += "%s %s\n" % (np.min(gridXCoords)-offsetX, np.max(gridXCoords)-offsetX)	# (xMin-offsetX) (xMax-offsetX)
	header += "%s %s\n" % (np.min(gridYCoords)-offsetY, np.max(gridYCoords)-offsetY)	# (yMin-offsetY) (yMax-offsetY)
	header += "%s %s" % (interpolZ.min(), interpolZ.max())	# zmin zMax		(allz.min(), allz.max())	# zmin zMax	

	if outFileName == None:
		if interp:
			outFileNameF = 'output_xyz2surfer_grd_'+interpType
		else:
			outFileNameF = 'output_xyz2surfer_grd_noInterpolation'
	else:
		outFileNameF = outFileName

	# sort z grid mesh so that each row only contains 10 numbers
	"""
	grdOutList = []
	for j in range(len(gridYCoords)):

		#check = 0
		
		lenRowTempMax = len(interpolZ[j])
		tempRowStr=''
		for loopN in range(lenRowTempMax):
			tempRowStr += str(interpolZ[j][loopN])+' '
		grdOutList.append(tempRowStr+'\n')
		#grdOutList.append('\n')

		
		rowId = 0
		lenRowTempMax = len(gridXCoords) #len(interpolZ[j])
		while lenRowTempMax > 0:

			#print(interpolZ[j])
			
			tempRowStr = ''
			if min([10,lenRowTempMax]) == 10:
				for loopN in range(rowId,rowId+10):
					#check+=1

					tempRowStr += str(float(interpolZ[j][loopN]))+' '
					#check += 1

				grdOutList.append(tempRowStr+'\n')
				rowId += 10

			elif min([10,lenRowTempMax]) == lenRowTempMax:
				#print(lenRowTemp)
				for loopN in range(rowId,rowId+lenRowTempMax):

					tempRowStr += str(float(interpolZ[j][loopN]))+' '
					#check += 1
			
				grdOutList.append(tempRowStr+'\n')
				grdOutList.append('\n')

			lenRowTempMax -= 10
		
			#print(check)
			#print(tempRowStr)
		#print(len(interpolZ[j]))

		#if lenRowTemp != len(tempRowStr.split('\n').split(' ')):
		#	print('this times it is weird %i %i' %(j,len(tempRowStr.split('\n').split(' '))))
	
	#print(grdOutList)
	"""
	# output file
	np.savetxt(outFileNameF+'.grd', interpolZ, header=header, fmt="%1.2f", comments='')
	"""
	with open(outFileNameF+'.grd', 'w', newline='') as grdfile:
		grdfile.write(header)
		for grdLine in grdOutList:
			grdfile.write(grdLine)
	"""

	#np.savetxt(outFileNameF+'.grd', grdOutList, header=header, comments='')#, fmt="%1.2f")

	return None

#################################################################################################################
### generate walls and modify DEM
#################################################################################################################
def generate_wall_dict_v1_00(wall_info):
	'''
	generate wall_dict containing all objects to be categorized as buildings or barriers

	# P or V 
	wall_info = [
		[wall_group_id, type ('P' or 'V'), slit_ratio, wall_segment_number, wall_segment_oriP (-90 ~ 90), wall_oriP (-90 ~ 90), 
		thickness, length, Z_opt (1~4), height/elevation, central_X_coord, central_Y_coord], ...
	]
	wall_data = [ [type ('P' or 'V'), wall_seg_id, Z_opt, height/elevation, [4 (corner X, Y)] ], ... ]

	# C 
	wall_info = [ [wall_group_id, type ('C'), cylinder_number, wall_oriP (-90 ~ 90), radius, length, Z_opt (1~4), height/elevation, central_X_coord, central_Y_coord], ... ]
	wall_data = [  [Type('C'), wall_seg_id, Z_opt, height/elevation, (central_X_coord, central_Y_coord, radius) ], ...	]

	# BD 
	wall_info = [ [wall_group_id, Type('BD'), Z_opt, height/elevation, XY_list ], ...] 
	wall_data = [ [Type('BD'), wall_seg_id, Z_opt, height/elevation, [(X,Y), ... from XY_list] ], ... ]
	
	# wall_dict -> key = type_wall_id, value = [(overall) wall_info, [each wall section data], [each wall section shapely polygon], sheply all wall multipolygon]
	
	for type_wall_id -> 
		ten thousands (0X,XXX) = type (10,000 = closed, 20,000 = P, 30,000 = V, 40,000 = C, 50,000 = BD)
		thousands and hundreds (X0,0XX) = wall number
		tens and ones (XX,X00) = wall segment number
	'''

	wall_dict = {}

	## add wall_info, wall_segment_data and wall_segment polygon into wall_dict
	for wall_i in wall_info:

		P_or_V_or_C_or_BD = wall_i[1]

		# compute number of wall segments
		# wall/box type
		if P_or_V_or_C_or_BD in ['P', 'V']:

			wall_group_id, P_or_V_type, slit_ratio, wall_segment_number, wall_segment_oriP, wall_oriP, thickness, length, Z_opt, h_or_z, central_X_coord, central_Y_coord = wall_i
			oriP_rad = np.deg2rad(wall_oriP)

			# single closed type
			if slit_ratio == 0 or wall_segment_number == 1:

				# type_wall_id
				type_wall_id = int(10_000 + wall_group_id*100)

				# 4 cornder XY coordinates
				corner_p1_x = central_X_coord - 0.5*np.sqrt(length**2 + thickness**2)*np.cos(oriP_rad + np.arctan(thickness/length))
				corner_p1_y = central_Y_coord - 0.5*np.sqrt(length**2 + thickness**2)*np.sin(oriP_rad + np.arctan(thickness/length))

				corner_p2_x = central_X_coord - 0.5*np.sqrt(length**2 + thickness**2)*np.cos(oriP_rad - np.arctan(thickness/length))
				corner_p2_y = central_Y_coord - 0.5*np.sqrt(length**2 + thickness**2)*np.sin(oriP_rad - np.arctan(thickness/length))

				corner_p3_x = central_X_coord + 0.5*np.sqrt(length**2 + thickness**2)*np.cos(oriP_rad + np.arctan(thickness/length))
				corner_p3_y = central_Y_coord + 0.5*np.sqrt(length**2 + thickness**2)*np.sin(oriP_rad + np.arctan(thickness/length))

				corner_p4_x = central_X_coord + 0.5*np.sqrt(length**2 + thickness**2)*np.cos(oriP_rad - np.arctan(thickness/length))
				corner_p4_y = central_Y_coord + 0.5*np.sqrt(length**2 + thickness**2)*np.sin(oriP_rad - np.arctan(thickness/length))

				corner_xy = [(corner_p1_x, corner_p1_y), (corner_p2_x, corner_p2_y), (corner_p3_x, corner_p3_y), (corner_p4_x, corner_p4_y)]

				# define shapely polygon
				wall_segment_polygon = Polygon(corner_xy)

				if type_wall_id not in wall_dict.keys():

					# type_wall_seg_id and type_wall_id
					type_wall_seg_id = int(10_000 + wall_group_id*100 + 1)

					# store to wall_dict
					wall_dict[type_wall_id] = [[deepcopy(wall_i)], [[P_or_V_type, type_wall_seg_id, Z_opt, h_or_z, corner_xy]], [wall_segment_polygon], None]

				else:

					temp_list = deepcopy(wall_dict[type_wall_id])

					# type_wall_seg_id and type_wall_id
					type_wall_seg_id = int( 10_000 + wall_group_id*100 + len(temp_list[2]) + 1 )

					temp_list[0].append(wall_i)
					temp_list[1].append([P_or_V_type, type_wall_seg_id, Z_opt, h_or_z, corner_xy])
					temp_list[2].append(wall_segment_polygon)

					# store to wall_dict
					wall_dict[type_wall_id] = deepcopy(temp_list)
					del temp_list

			# slit walls
			else:

				# type_wall_id
				if P_or_V_type == 'P': # all equal to the value
					type_wall_id = int(20_000 + wall_group_id*100)
				elif P_or_V_type == 'V': # alternating between + (set angle) and - (negative of set angle) orientation angle
					type_wall_id = int(30_000 + wall_group_id*100)

				# spacing  
				spacing = slit_ratio*length/(wall_segment_number - 1)

				# starting point
				starting_centroid_X = central_X_coord - (0.5*length)*np.cos(oriP_rad)
				starting_centroid_Y = central_Y_coord - (0.5*length)*np.sin(oriP_rad)

				# wall segment length
				wall_length = length*(1-slit_ratio)/wall_segment_number

				# create list to store info and set type_wall_seg_id numbers
				if type_wall_id not in wall_dict.keys(): 
					temp_list = [[deepcopy(wall_i)], [], [], None]

				else:
					temp_list = deepcopy(wall_dict[type_wall_id])
					temp_list[0].append(wall_i)

				# add each wall segment information and shapely polygon
				for wall_seg_n in range(wall_segment_number):

					# leftmost-downmost cornder XY coordinates of the wall segment
					wall_segment_centroid_X = (0.5*wall_length + wall_seg_n*(spacing + wall_length))*np.cos(oriP_rad) + starting_centroid_X
					wall_segment_centroid_Y = (0.5*wall_length + wall_seg_n*(spacing + wall_length))*np.sin(oriP_rad) + starting_centroid_Y

					# wall segment data
					if P_or_V_type == 'P': # all equal to the value

						# wall segment orientation
						wall_seg_oriP_rad = oriP_rad + np.deg2rad(wall_segment_oriP)

						# type_wall_seg_id 
						type_wall_seg_id = int(20_000 + wall_group_id*100 + (len(temp_list[1]) + 1) + wall_seg_n)

					elif P_or_V_type == 'V': # alternating between + (set angle) and - (negative of set angle) orientation angle
						
						# wall segment orientation
						wall_seg_oriP_rad = oriP_rad + ((-1)**wall_seg_n)*np.deg2rad(wall_segment_oriP)

						# type_wall_seg_id 
						type_wall_seg_id = int(30_000 + wall_group_id*100 + (len(temp_list[1]) + 1) + wall_seg_n)

					# leftmost-downmost cornder XY coordinates
					corner_p1_x = wall_segment_centroid_X - 0.5*np.sqrt(wall_length**2 + thickness**2)*np.cos(wall_seg_oriP_rad + np.arctan(thickness/wall_length))
					corner_p1_y = wall_segment_centroid_Y - 0.5*np.sqrt(wall_length**2 + thickness**2)*np.sin(wall_seg_oriP_rad + np.arctan(thickness/wall_length))

					corner_p2_x = wall_segment_centroid_X - 0.5*np.sqrt(wall_length**2 + thickness**2)*np.cos(wall_seg_oriP_rad - np.arctan(thickness/wall_length))
					corner_p2_y = wall_segment_centroid_Y - 0.5*np.sqrt(wall_length**2 + thickness**2)*np.sin(wall_seg_oriP_rad - np.arctan(thickness/wall_length))

					corner_p3_x = wall_segment_centroid_X + 0.5*np.sqrt(wall_length**2 + thickness**2)*np.cos(wall_seg_oriP_rad + np.arctan(thickness/wall_length))
					corner_p3_y = wall_segment_centroid_Y + 0.5*np.sqrt(wall_length**2 + thickness**2)*np.sin(wall_seg_oriP_rad + np.arctan(thickness/wall_length))

					corner_p4_x = wall_segment_centroid_X + 0.5*np.sqrt(wall_length**2 + thickness**2)*np.cos(wall_seg_oriP_rad - np.arctan(thickness/wall_length))
					corner_p4_y = wall_segment_centroid_Y + 0.5*np.sqrt(wall_length**2 + thickness**2)*np.sin(wall_seg_oriP_rad - np.arctan(thickness/wall_length))

					corner_xy = [(corner_p1_x, corner_p1_y), (corner_p2_x, corner_p2_y), (corner_p3_x, corner_p3_y), (corner_p4_x, corner_p4_y)]

					# define shapely polygon
					wall_segment_polygon = Polygon(corner_xy)

					# store to all wall dict temp list
					temp_list[1].append([P_or_V_type, type_wall_seg_id, Z_opt, h_or_z, corner_xy])
					temp_list[2].append(wall_segment_polygon)

				## store to all wall dict 
				wall_dict[type_wall_id] = deepcopy(temp_list)
				del temp_list

		# circular
		elif P_or_V_or_C_or_BD == 'C':

			wall_group_id, C_type, cylinder_number, wall_oriP, radius, length, Z_opt, h_or_z, central_X_coord, central_Y_coord = wall_i
			oriP_rad = np.deg2rad(wall_oriP)

			# type_wall_id
			type_wall_id = int(40_000 + wall_group_id*100)

			# spacing  
			spacing = (length - 2*radius*wall_segment_number)/(wall_segment_number - 1)

			# starting center point
			starting_center_X = central_X_coord - (0.5*length - radius)*np.cos(oriP_rad)
			starting_center_Y = central_Y_coord - (0.5*length - radius)*np.sin(oriP_rad)

			# create list to store info and set type_wall_seg_id numbers
			if type_wall_id not in wall_dict.keys(): 
				temp_list = [ [deepcopy(wall_i)], [], [], None]

			else:
				temp_list = deepcopy(wall_dict[type_wall_id])
				temp_list[0].append(wall_i)

			# add each wall segment information and shapely polygon
			for wall_seg_n in range(wall_segment_number):

				# type_wall_seg_id
				type_wall_seg_id = int(40_000 + wall_group_id*100 + (len(temp_list[1]) + 1) + wall_seg_n)

				# leftmost-downmost cornder XY coordinates
				center_x = wall_seg_n*(spacing + 2*radius)*np.cos(oriP_rad) + starting_center_X
				center_y = wall_seg_n*(spacing + 2*radius)*np.sin(oriP_rad) + starting_center_Y

				# cylinder shapely polygon
				cylinder_polygon = Point(center_x, center_y).buffer(radius) 

				# store to all wall dict temp list
				temp_list[1].append([C_type, type_wall_seg_id, Z_opt, h_or_z, (center_x, center_y, radius)])
				temp_list[2].append(cylinder_polygon)
			
			## store to all wall dict 
			wall_dict[type_wall_id] = deepcopy(temp_list)
			del temp_list

		# general building - any shape
		elif P_or_V_or_C_or_BD == 'BD': 

			wall_group_id, BD_type, Z_opt, h_or_z, XY_list = wall_i

			# define shapely polygon
			XY_tuple_list = [(xy[0],xy[1]) for xy in XY_list]
			BD_polygon = Polygon(XY_tuple_list)

			# type_wall_id
			type_wall_id = int( 50_000 + wall_group_id*100 )

			if type_wall_id not in wall_dict.keys():

				# type_wall_seg_id and type_wall_id
				type_wall_seg_id = int(50_000 + wall_group_id*100 + 1)

				# store to wall_dict
				wall_dict[type_wall_id] = [[deepcopy(wall_i)], [[BD_type, type_wall_seg_id, Z_opt, h_or_z, deepcopy(XY_tuple_list)]], [BD_polygon], None]

			else:

				temp_list = deepcopy(wall_dict[type_wall_id])

				# type_wall_seg_id and type_wall_id
				type_wall_seg_id = int( 50_000 + wall_group_id*100 + len(temp_list[2]) + 1 )

				temp_list[0].append(wall_i)
				temp_list[1].append([P_or_V_type, type_wall_seg_id, Z_opt, h_or_z, deepcopy(XY_tuple_list)])
				temp_list[2].append(BD_polygon)

				# store to wall_dict
				wall_dict[type_wall_id] = deepcopy(temp_list)
				del temp_list

	## multipolygon - contains all the wall segment shapely polygon - for each wall_num
	for wall_dict_key in wall_dict.keys():
		
		temp_list = deepcopy(wall_dict[wall_dict_key])

		temp_list[3] = MultiPolygon(temp_list[2])

		# store to wall_dict
		wall_dict[wall_dict_key] = deepcopy(temp_list)
		del temp_list

	return wall_dict

# modify DEM file to contain walls 
def modify_DEM_v1_0(topoFileName, wall_segement_data, wall_poly, outFileName=None, saveFileFormat=None):
	'''
	modify DEM file to contain walls like DTM

	# input
		topoFileName	:	input topo file name (if external file, csv or grd file)
		Note: if external file, it must include the file format identity. e.g. if csv include '.csv'
			file format (csv)
				col 1 = x
				col 2 = y
				col 3 = z

		wall_segement_data	:	input wall info
			col 1 = type (P, V, C, BD) 	
			col 2 = type_wall_seg_id	
			col 3 = Z_opt  
			col 4 = height/elevation 
			col 5 = XY points or (X,Y,R)	

			key:	Z-opt = option for calculaing z coordinates
						1 = equal z value for the whole wall (z-coordinate = elevation)
						2 = equal addition of z value on top of ground z coordinate value (z-coordinate = topo z-value + height)
						3 = equal for whole wall to the minimum z value on top  of ground z coordinate value (z-coordinate = min(topo z-value + height))
						4 = equal for whole wall to the maximum z value on top  of ground z coordinate value (z-coordinate = max(topo z-value + height))

		wall_poly : shapely polygons

		outFileName		:	output file name (file)

		saveFileFormat	:	save the new file into specified file format (default = None)
			None = no output saved - only dictionary and modified XYZ
			'csv' = xyz csv file
			'grd' = surfer grd file  

	# output
		outFileName		:	output xyz file name  
	'''

	# sort topo file - import topo file into xyz np.array file
	topoXYZ = np.array(csv2list(topoFileName))
	
	## extract bd input data
	# used to find the elevation
	z_opt = wall_segement_data[2]		# 1 = given Z;  2 = deltaZ ; 3 = min Z computed; 4 = max Z computed
	b_height = wall_segement_data[3]	# height/elevation

	## find exterior points of the wall - plan view
	wall_poly_ext_xy = list(wall_poly.exterior.coords)

	## find all the topo points that within the wall perimeter 
	# source: 'https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python'
	XYpoints = topoXYZ[:,(0,1)]					# DEM XY coordinates
	p = Path(wall_poly_ext_xy) 					# make a path polygon from the wall polygon exterior XY points
	gridBool = p.contains_points(XYpoints)		# boolean showing all the points within the wall polygon

	# iterate and find all the points
	indexPinWall = []
	for indexN, boolL in enumerate(gridBool):
		if boolL:
			indexPinWall.append(indexN)

	## find new z coordinates
	newTopowXYZ = []
	for loopPinWall in indexPinWall:
		# check the elevation computation option (Z-opt)
		if z_opt == 1:	# 1 = same Z
			newTopowXYZ.append([topoXYZ[loopPinWall][0], topoXYZ[loopPinWall][1], b_height])
		elif z_opt in [2,3,4]: # 2 = same deltaZ; 3 = min Z computed; 4 = max Z computed
			newTopowXYZ.append([topoXYZ[loopPinWall][0], topoXYZ[loopPinWall][1], topoXYZ[loopPinWall][2]+b_height])

	newTopowXYZ = np.array(newTopowXYZ)
	# for z-opt equal to 3 or 4 additional process is taken
	if z_opt == 3: # z-opt = 3 - min
		minZcomp = newTopowXYZ[:,2].min()
		newTopowXYZ[:,2] = minZcomp

	elif z_opt == 4:  # z-opt = 4 - max
		maxZcomp = newTopowXYZ[:,2].max()
		newTopowXYZ[:,2] = maxZcomp

	## create new topography grid with the wall data included
	topoXYZwithWall = deepcopy(topoXYZ)

	for idxloop, xyzloop in zip(indexPinWall, newTopowXYZ):
		topoXYZwithWall[idxloop] = xyzloop

	## export data
	# output
	if saveFileFormat == 'csv': 	# csv file
		exportList2CSV(outFileName+'.csv', topoXYZwithWall)

	elif saveFileFormat == 'grd':	# grd file
		xyz2grd(topoXYZwithWall, offset=None, outFileName=outFileName, interp=False)

	# return topoXYZwithWall
	return None

#################################################################################################################
### SPEC-debris - classes
#################################################################################################################
# clusterID, particles, time, row, col, xc, yc, zc, sc, Vc, uc, hc, predecessor, Dc, Pc, ll, merged, dfl_type, dfl_id
# input = clusterID, part_list, predecessor
class Cluster:

	def __init__(self, clusterID, part_list, predecessor):

		self.clusterID = clusterID  # cluster ID number
		self.particles = part_list  # [list of class Particle]
		self.predecessor = predecessor  # previous cluster data

		self.time = 0               # time step
		self.sc = 0              # centroid cumulative travel distance
		
		self.xc = 0              # centroid x-coordinate
		self.yc = 0              # centroid y-coordinate
		self.zc = 0              # centroid z-coordinate

		self.uc = 0              # cluster average speed (velocity magnitude)
		self.hc = 0              # cluster interpolated depth

		self.Vc = 0              # cluster total volume
		
		self.Dc = 0              # cluster travel distance
		
		self.Pc = 0              # cluster pressure
		self.Fr = 0 			 # Froude number = v/sqrt(g*h)
		
		self.merged = 0          # 0 = False, 1 = True
		self.dfl_type = 0
		self.dfl_id = None

		self.concave_hull_alpha = 0  # alpha value for concave hull (if alpha = 0, concave hull == convex hull)
		self.boundary_pt_xy = None 		 # cluster boundary XY coordinates
		self.boundary_polygon = None 	 # cluster boundary shapely polygon

		self.area = 0 					# cluster area

	def __str__(self):
		return 'cID:%i, t:%0.1f, s:%0.2f, X: %0.2f, Y:%0.2f, Z:%0.3f, vel:%0.2f, dep:%0.2f, Vol:%0.2f, Dist:%0.2f, Pres:%0.1f' % (self.clusterID, self.time, self.sc, self.xc, self.yc, self.zc, self.uc, self.hc, self.Vc, self.Dc, self.Pc)

	# compute centroid location xy using mean particle positions
	def compute_centroid(self):
		all_part_xy_list = [(part.x,part.y) for part in self.particles]
		all_part_xy_array = np.array(all_part_xy_list)
		mean_xy = np.mean(all_part_xy_array, axis=0)
		self.xc = mean_xy[0]
		self.yc = mean_xy[1]
		return (self.xc, self.yc)

	# compute centroid location xy using median particle positions
	def compute_median_centroid(self):
		all_part_xy_list = [(part.x,part.y) for part in self.particles]
		all_part_xy_array = np.array(all_part_xy_list)
		median_xy = np.median(all_part_xy_array, axis=0)
		self.xc = median_xy[0]
		self.yc = median_xy[1]
		return (self.xc, self.yc)

	# max pressure centroid
	def compute_max_pressure_centroid(self):
		# all_part_xy_listst = [(part.x,part.y) for part in self.particles]
		all_part_p_list = [part.Pi for part in self.particles]
		max_p_idx = all_part_p_list.index(max(all_part_p_list))
		self.xc = self.particles[max_p_idx].x
		self.yc = self.particles[max_p_idx].y
		return (self.xc, self.yc)


	# insert elevation if already computed
	def insert_Z(self, Z):
		self.z = Z

	# compute cluster area
	def compute_area(self):
		boundary_poly = self.boundary_polygon
		self.area = boundary_poly.area
		return self.area

	# compute cumulative travel distance (s)
	def compute_s(self):
		self.sc = self.predecessor.sc + np.sqrt((self.xc-self.predecessor.xc)**2 + (self.yc-self.predecessor.yc)**2 + (self.zc-self.predecessor.zc)**2)
		return self.sc

	# compute distance from road (D)
	# road_xy_list = [(goal1[0], goal1[1]), (goal2[0], goal2[1])]
	def compute_D(self, road_xy_list):
		xr1 = road_xy_list[0][0]
		yr1 = road_xy_list[0][1]
		xr2 = road_xy_list[1][0]
		yr2 = road_xy_list[1][1]
		denominator = np.sqrt((yr1-yr2)**2 + (xr1-xr2)**2)
		numerator = (yr2-yr1)*self.xc - (xr2-xr1)*self.yc + xr2*yr1 - xr1*yr2
		self.Dc = abs(numerator)/denominator
		return self.Dc

	def compute_P(self, g):
		part_av_density = np.mean([part.rho for part in self.particles])
		if self.uc > 0:
			self.Pc = (part_av_density/1000)*(0.5*g*self.hc + (self.uc)**2)  # kPa
		else: # hydrostatic pressure
			self.Pc = (part_av_density/1000)*g*self.hc  # kPa
		return self.Pc

	def compute_alpha_P(self, alpha):
		part_av_density = np.mean([part.rho for part in self.particles])
		self.Pc = alpha*(part_av_density/1000)*((self.uc)**2)  # kPa
		return self.Pc

	def compute_Fr(self, g):
		self.Fr = self.uc/np.sqrt(g*self.hc)
		return self.Fr

	def compute_Fr_P(self, g):
		part_av_density = np.mean([part.rho for part in self.particles])

		# https://onlinelibrary.wiley.com/doi/full/10.1002/esp.3744
		if self.uc > 0:
			self.Fr = self.compute_Fr(g)
			self.Pc = 5.3*(self.Fr**(-1.5))*(part_av_density/1000)*((self.uc)**2)  # kPa
		else: # hydrostatic pressure
			self.Pc = (part_av_density/1000)*g*self.hc  # kPa
		return self.Pc

	# compute cumnulative cluster volume
	def compute_V(self):
		Vi_array = np.array([part.Vi for part in self.particles])
		self.Vc = np.sum(Vi_array)
		return self.Vc

	def compute_av_h_u(self):
		av_velocity = np.mean(np.array([part.ui for part in self.particles]))
		av_depth = np.mean(np.array([part.hi for part in self.particles]))

		self.hc = av_depth
		self.uc = av_velocity

		return av_depth, av_velocity

	def compute_median_h_u(self):
		median_velocity = np.median(np.array([part.ui for part in self.particles]))
		median_depth = np.median(np.array([part.hi for part in self.particles]))

		self.hc = median_depth
		self.uc = median_velocity

		return median_depth, median_velocity

	# update time_step
	def update_time(self, d_time=1):
		self.time = self.predecessor.time + d_time

	# update merging
	def update_merging(self):
		self.merged = True

	## return all values
	# return all cluster data
	def return_all(self):
		return [self.clusterID, self.time, self.sc, self.xc, self.yc, self.zc, self.uc, self.hc, self.Vc, self.Dc, self.Pc, self.area, self.Fr, self.concave_hull_alpha, self.merged]

	def return_all_opt(self):
		return [self.clusterID, self.time, self.sc, self.xc, self.yc, self.zc, self.uc, self.Vc, self.Dc, self.Pc, self.merged]

# X, Y, Z, Vi, ui, hi, radius, clusterID, materialID, predecessor
class Particle:

	def __init__(self, X, Y, Z, Vi, ui, hi, radius, clusterID, materialID, predecessor):
		
		self.predecessor = predecessor  # predecessor particle class
		self.clusterID = clusterID  # integer ID of initial source location
		
		self.time = 0    # time step
		self.si = 0      # cumulative travel distance from the source

		# XYZ coordinates
		self.x = X
		self.y = Y
		self.z = Z  	

		# particle radius
		self.r = radius

		# material properties
		self.materialID = materialID  	# material ID
		self.phi = None 				# internal friction angle
		self.fb = None 					# basal resistance - Voellmy rheology
		self.ft = None 					# turbulance resistance - Voellmy rheology
		self.rho = None 				# density
		self.Es = None 					# erosion growth rate

		# dip and dip direction
		self.dip = 0
		self.dip_direction = 0
  
		# travel direction 
		self.travel_direction = 0
		self.cross_travel_direction = 0

		# gradient in X and Y direction
		self.gradients_x = 0
		self.gradients_y = 0
		self.dip_travel_direction = 0
		self.cross_dip_travel_direction = 0

		self.Vi = Vi    # volume

		self.k_MD = 1 
		self.k_XMD = 1

		self.k_MD_max = 1 
		self.k_MD_min = 1
		self.k_XMD_max = 1 
		self.k_XMD_min = 1
  
		self.ll = 4*radius
		self.hi = hi    # depth
		self.div_hi = 0
  
		self.dhdxi = 1
		self.dhdyi = 1
  
		self.dh_dMD = 0
		self.dh_dXMD = 0

		self.ui = ui    # depth-averaged velocity magnitude
		self.ux = None  # depth-averaged velocity in x-direction
		self.uy = None  # depth-averaged velocity in y-direction

		self.a_MD = 0  	# overall force in movement direction MD
		self.a_XMD = 0  # overall force in cross movement direction XMD
  
		self.dx_grad_local = 0
		self.dy_grad_local = 0
		
		self.Fr = 0 	 # Froude number = v/sqrt(g*h)
		self.Pi = 0      # pressure

		self.sigma = None 		# bed-normal stress
		self.tau_r = None 		# shear stress - rheology
  
		self.wall = False
		self.elevation = Z

	# def __eq__(self, other):
	# 	if self.x == other.x and self.y == other.y:
	# 		return True
	# 	return False

	def __str__(self):
		return 'ID:%i, t:%0.1f, s:%0.2f, X: %0.2f, Y:%0.2f, Z:%0.3f, Vol:%0.2f, vel:%0.2f, dep:%0.2f' % (self.clusterID, self.time, self.si, self.x, self.y, self.z, self.Vi, self.ui, self.hi)

	## insert new values into the particles 
	# insert elevation if already computed
	def insert_Z(self, Z):
		self.z = Z

	# insert clusterID after sorted
	def insert_cID(self, cID):
		self.clusterID = cID

	# replace depth
	def insert_h(self, new_h):
		self.hi = new_h

	# replace material properties
	def insert_material(self, material_dict):
		self.phi = material_dict[self.materialID]["phi"]
		self.fb = material_dict[self.materialID]["f"][0]
		self.ft = material_dict[self.materialID]["f"][1]
		self.rho = material_dict[self.materialID]["density"]
		self.Es = material_dict[self.materialID]["Es"]

	## compute value based on the data
	# compute cumulative travel distance (s)
	def compute_s(self):
		self.si = self.predecessor.si + np.sqrt((self.x-self.predecessor.x)**2 + (self.y-self.predecessor.y)**2 + (self.z-self.predecessor.z)**2)
		return self.si

	# compute depth gradient - SPH
	def compute_h_local_dh_check_Es(self, other_part_list, material_dict, erode_DEM_ij, ll):
		'''
		V_array :	array of neighboring particles' volume
		s_array :	array of Euclidian 3D distance with neighboring particles
		ll 		:	smoothing length 
		'''
		# SPH - smoothing length
		self.ll = ll

		if len(other_part_list) > 1:
			
			## get current particle distance - S_cur
			# current particle XY 
			cur_other_parts_xy = [(part_c.x, part_c.y) for part_c in other_part_list ]
			cur_other_parts_xy_array = np.array(cur_other_parts_xy)
			cur_other_parts_xy_rotated = rotate_part_xy_local((self.x, self.y), cur_other_parts_xy, self.travel_direction)
			cur_other_parts_xy_rotated_array = np.array(cur_other_parts_xy_rotated)
   
			cur_other_parts_z_array = np.array([part_c.z for part_c in other_part_list])

			# compute distance in global frame
			dx_array = cur_other_parts_xy_array[:,0] - self.x
			dy_array = cur_other_parts_xy_array[:,1] - self.y
			dz_array = cur_other_parts_z_array - self.z

			# compute distance in reorientated frame
			dxp_array = cur_other_parts_xy_rotated_array[:,0] - self.x
			dyp_array = cur_other_parts_xy_rotated_array[:,1] - self.y

			# s_array = np.power( np.power(dxp_array,2) + np.power(dyp_array,2) , 0.5)
			s_array = np.power( np.power(dxp_array,2) + np.power(dyp_array,2) + np.power(dz_array,2) , 0.5)
			qq_array = s_array/ll
   
			## compute particle kernel - Gaussian
			a_g = 1/(np.pi*(ll**2))    # 2D

			W_array_t = np.absolute(a_g*np.exp(-1*np.power(qq_array, 2)))
			W_array = np.where(qq_array>2, 0.0, W_array_t)

			dW_array_t = np.absolute(a_g*2*qq_array*np.exp(-1*np.power(qq_array, 2)))
			dW_array = np.where(qq_array>2, 0.0, dW_array_t)
			
			# volume 
			V_list = [part_c.Vi for part_c in other_part_list]
			V_array = np.array(V_list)		

			V_old_array = np.array([part_c.predecessor.Vi for part_c in other_part_list])
			dV_array = V_array - V_old_array

			# compute depth and depth gradient
			np.seterr(invalid='ignore')
			self.hi = np.nansum(W_array*V_array)
			
			self.dhdxi = np.nansum(dW_array*V_array*(dx_array/ll)/np.sqrt(np.power(dx_array, 2)+np.power(dy_array, 2)))
			self.dhdyi = np.nansum(dW_array*V_array*(dy_array/ll)/np.sqrt(np.power(dx_array, 2)+np.power(dy_array, 2)))

			self.dh_dMD = np.nansum(dW_array*V_array*(dxp_array/ll)/np.sqrt(np.power(dxp_array, 2)+np.power(dyp_array, 2)))
			self.dh_dXMD = np.nansum(dW_array*V_array*(dyp_array/ll)/np.sqrt(np.power(dxp_array, 2)+np.power(dyp_array, 2)))
			erode_depth = np.nansum(W_array*dV_array)  # compute eroded depth
	
			# cumulation of eroding depth
			if (self.Vi - self.predecessor.Vi) > 0:
				erode_DEM_ij += erode_depth
			elif (self.Vi - self.predecessor.Vi) < 0:
				erode_DEM_ij -= erode_depth

			# deposition
			if erode_DEM_ij <= 0:
				erode_DEM_ij = 0
			# reached max erosion depth
			elif erode_DEM_ij >= material_dict[self.materialID]["max erode depth"]:
				erode_DEM_ij = material_dict[self.materialID]["max erode depth"]
				self.Es = 0

		elif len(other_part_list) <= 1:
			
			# Gaussian Kernel   
			self.hi = self.Vi/(np.pi*(ll**2))
			self.dhdxi = 0
			self.dhdyi = 0
			self.dh_dMD = 0
			self.dh_dXMD = 0

			erode_depth = (self.Vi - self.predecessor.Vi)/(np.pi*(ll**2))
			erode_DEM_ij += erode_depth

		return self.hi, self.dh_dMD, self.dh_dXMD, erode_DEM_ij

	def compute_h_local_dh_check_Es_ghost(self, other_part_ghost_list, material_dict, erode_DEM_ij, ll):
		'''
		V_array :	array of neighboring particles' volume
		s_array :	array of Euclidian 3D distance with neighboring particles
		ll 		:	smoothing length 
		'''
		# SPH - smoothing length
		self.ll = ll
  
		if len(other_part_ghost_list) > 1:
			
			## get current particle distance - S_cur
			# current particle XY 
			cur_other_parts_xy = [(part_c.x, part_c.y) for part_c in other_part_ghost_list ]
			cur_other_parts_xy_array = np.array(cur_other_parts_xy)
			cur_other_parts_xy_rotated = rotate_part_xy_local((self.x, self.y), cur_other_parts_xy, self.travel_direction)
			cur_other_parts_xy_rotated_array = np.array(cur_other_parts_xy_rotated)
   
			cur_other_parts_z_array = np.array([part_c.z if isinstance(part_c.z, (int, float)) else self.z for part_c in other_part_ghost_list])

			# compute distance in global frame
			dx_array = cur_other_parts_xy_array[:,0] - self.x
			dy_array = cur_other_parts_xy_array[:,1] - self.y
			dz_array = cur_other_parts_z_array - self.z

			# compute distance in reorientated frame
			dxp_array = cur_other_parts_xy_rotated_array[:,0] - self.x
			dyp_array = cur_other_parts_xy_rotated_array[:,1] - self.y

			# s_array = np.power( np.power(dxp_array,2) + np.power(dyp_array,2) , 0.5)
			s_array = np.power( np.power(dxp_array,2) + np.power(dyp_array,2) + np.power(dz_array,2) , 0.5)
			qq_array = s_array/ll
   
			## compute particle kernel - Gaussian
			a_g = 1/(np.pi*(ll**2))    # 2D

			W_array_t = np.absolute(a_g*np.exp(-1*np.power(qq_array, 2)))
			W_array = np.where(qq_array>2, 0.0, W_array_t)

			dW_array_t = np.absolute(a_g*2*qq_array*np.exp(-1*np.power(qq_array, 2)))
			dW_array = np.where(qq_array>2, 0.0, dW_array_t)
			
			# volume 
			V_list = [part_c.Vi for part_c in other_part_ghost_list]
			V_array = np.array(V_list)		

			V_old_array = np.array([part_c.predecessor.Vi if isinstance(part_c.predecessor.Vi, (int, float)) else part_c.Vi for part_c in other_part_ghost_list])
			dV_array = V_array - V_old_array

			# compute depth and depth gradient
			np.seterr(invalid='ignore')
			self.hi = np.nansum(W_array*V_array)
			self.div_hi = np.nansum(dW_array*V_array)
			
			self.dhdxi = np.nansum(dW_array*V_array*(dx_array/ll)/np.sqrt(np.power(dx_array, 2)+np.power(dy_array, 2)))
			self.dhdyi = np.nansum(dW_array*V_array*(dy_array/ll)/np.sqrt(np.power(dx_array, 2)+np.power(dy_array, 2)))

			self.dh_dMD = np.nansum(dW_array*V_array*(dxp_array/ll)/np.sqrt(np.power(dxp_array, 2)+np.power(dyp_array, 2)))
			self.dh_dXMD = np.nansum(dW_array*V_array*(dyp_array/ll)/np.sqrt(np.power(dxp_array, 2)+np.power(dyp_array, 2)))
			erode_depth = np.nansum(W_array*dV_array)  # compute eroded depth
	
			# cumulation of eroding depth
			if (self.Vi - self.predecessor.Vi) > 0:
				erode_DEM_ij += erode_depth
			elif (self.Vi - self.predecessor.Vi) < 0:
				erode_DEM_ij -= erode_depth

			# deposition
			if erode_DEM_ij <= 0:
				erode_DEM_ij = 0
			# reached max erosion depth
			elif erode_DEM_ij >= material_dict[self.materialID]["max erode depth"]:
				erode_DEM_ij = material_dict[self.materialID]["max erode depth"]
				self.Es = 0

		elif len(other_part_ghost_list) <= 1:
			
			# Gaussian Kernel   
			self.hi = self.Vi/(np.pi*(ll**2))
			self.div_hi = 0
			self.dhdxi = 0
			self.dhdyi = 0
			self.dh_dMD = 0
			self.dh_dXMD = 0

			erode_depth = (self.Vi - self.predecessor.Vi)/(np.pi*(ll**2))
			erode_DEM_ij += erode_depth

		# return self.hi, self.div_hi, self.dh_dMD, self.dh_dXMD, erode_DEM_ij
		return erode_DEM_ij

	def compute_h_local_dh_check_Es_ghost0(self, other_part_list, other_part_ghost_list, material_dict, erode_DEM_ij, ll):
		'''
		V_array :	array of neighboring particles' volume
		s_array :	array of Euclidian 3D distance with neighboring particles
		ll 		:	smoothing length 
		'''
		# SPH - smoothing length
		self.ll = ll
		
		## for depth - include ghost particles
		if len(other_part_ghost_list) > 1:
			
			## get current particle distance - S_cur
			# current particle XY 
			cur_other_parts_xyg = [(part_c.x, part_c.y) for part_c in other_part_ghost_list ]
			cur_other_parts_xy_arrayg = np.array(cur_other_parts_xyg)
			cur_other_parts_z_arrayg = np.array([part_c.z if isinstance(part_c.z, (int, float)) else self.z for part_c in other_part_ghost_list])

			# compute distance in global frame
			dxg_array = cur_other_parts_xy_arrayg[:,0] - self.x
			dyg_array = cur_other_parts_xy_arrayg[:,1] - self.y
			dzg_array = cur_other_parts_z_arrayg - self.z

			# s_array = np.power( np.power(dxp_array,2) + np.power(dyp_array,2) , 0.5)
			sg_array = np.power( np.power(dxg_array,2) + np.power(dyg_array,2) + np.power(dzg_array,2) , 0.5)
			qqg_array = sg_array/ll
   
			## compute particle kernel - Gaussian
			a_g = 1/(np.pi*(ll**2))    # 2D

			Wg_array_t = np.absolute(a_g*np.exp(-1*np.power(qqg_array, 2)))
			Wg_array = np.where(qqg_array>2, 0.0, Wg_array_t)
			
			# volume 
			Vg_list = [part_c.Vi for part_c in other_part_ghost_list]
			Vg_array = np.array(Vg_list)		

			# compute depth and depth gradient
			np.seterr(invalid='ignore')
			self.hi = np.nansum(Wg_array*Vg_array)
   
		elif len(other_part_ghost_list) <= 1:
			# Gaussian Kernel   
			self.hi = self.Vi/(np.pi*(ll**2))

		## for depth gradient - do not include ghost particles
		if len(other_part_list) > 1:
				
			## get current particle distance - S_cur
			# current particle XY 
			cur_other_parts_xy = [(part_c.x, part_c.y) for part_c in other_part_list ]
			cur_other_parts_xy_array = np.array(cur_other_parts_xy)
			cur_other_parts_xy_rotated = rotate_part_xy_local((self.x, self.y), cur_other_parts_xy, self.travel_direction)
			cur_other_parts_xy_rotated_array = np.array(cur_other_parts_xy_rotated)
   
			cur_other_parts_z_array = np.array([part_c.z for part_c in other_part_list])

			# compute distance in global frame
			dx_array = cur_other_parts_xy_array[:,0] - self.x
			dy_array = cur_other_parts_xy_array[:,1] - self.y
			dz_array = cur_other_parts_z_array - self.z

			# compute distance in reorientated frame
			dxp_array = cur_other_parts_xy_rotated_array[:,0] - self.x
			dyp_array = cur_other_parts_xy_rotated_array[:,1] - self.y

			# s_array = np.power( np.power(dxp_array,2) + np.power(dyp_array,2) , 0.5)
			s_array = np.power( np.power(dxp_array,2) + np.power(dyp_array,2) + np.power(dz_array,2) , 0.5)
			qq_array = s_array/ll
   
			## compute particle kernel - Gaussian
			a_g = 1/(np.pi*(ll**2))    # 2D

			W_array_t = np.absolute(a_g*np.exp(-1*np.power(qq_array, 2)))
			W_array = np.where(qq_array>2, 0.0, W_array_t)

			dW_array_t = np.absolute(a_g*2*qq_array*np.exp(-1*np.power(qq_array, 2)))
			dW_array = np.where(qq_array>2, 0.0, dW_array_t)
			
			# volume 
			V_list = [part_c.Vi for part_c in other_part_list]
			V_array = np.array(V_list)		

			V_old_array = np.array([part_c.predecessor.Vi for part_c in other_part_list])
			dV_array = V_array - V_old_array

			# compute depth and depth gradient
			np.seterr(invalid='ignore')
			self.div_hi = np.nansum(dW_array*V_array)
			
			self.dhdxi = np.nansum(dW_array*V_array*(dx_array/ll)/np.sqrt(np.power(dx_array, 2)+np.power(dy_array, 2)))
			self.dhdyi = np.nansum(dW_array*V_array*(dy_array/ll)/np.sqrt(np.power(dx_array, 2)+np.power(dy_array, 2)))

			self.dh_dMD = np.nansum(dW_array*V_array*(dxp_array/ll)/np.sqrt(np.power(dxp_array, 2)+np.power(dyp_array, 2)))
			self.dh_dXMD = np.nansum(dW_array*V_array*(dyp_array/ll)/np.sqrt(np.power(dxp_array, 2)+np.power(dyp_array, 2)))
			erode_depth = np.nansum(W_array*dV_array)  # compute eroded depth
	
			# cumulation of eroding depth
			if (self.Vi - self.predecessor.Vi) > 0:
				erode_DEM_ij += erode_depth
			elif (self.Vi - self.predecessor.Vi) < 0:
				erode_DEM_ij -= erode_depth

			# deposition
			if erode_DEM_ij <= 0:
				erode_DEM_ij = 0
			# reached max erosion depth
			elif erode_DEM_ij >= material_dict[self.materialID]["max erode depth"]:
				erode_DEM_ij = material_dict[self.materialID]["max erode depth"]
				self.Es = 0

		elif len(other_part_list) <= 1:
			
			# Gaussian Kernel   
			self.dhdxi = 0
			self.dhdyi = 0
			self.dh_dMD = 0
			self.dh_dXMD = 0

			erode_depth = (self.Vi - self.predecessor.Vi)/(np.pi*(ll**2))
			erode_DEM_ij += erode_depth

		# return self.hi, self.div_hi, self.dh_dMD, self.dh_dXMD, erode_DEM_ij
		return erode_DEM_ij

	# compute dip, dip direction and gradients
	def compute_dip_and_angles(self, local_xy, local_z):

		# least-squares problem - get local gradient in X- and Y-direction
		local_xy_array = np.array(local_xy)
		local_z_array = np.array(local_z)

		reg_local = LinearRegression().fit(local_xy_array, local_z_array)
		DEM_gradients = reg_local.coef_

		# down-slope -> (+); up-slope -> (-)
		self.gradients_x = -DEM_gradients[0]
		self.gradients_y = -DEM_gradients[1]
  
		self.dip = abs(np.arccos( 1 / np.sqrt(1 + (self.gradients_x)**2 + (self.gradients_y)**2 )))
		self.dip_direction = np.arctan2(self.gradients_y, self.gradients_x)
		dip_direction_bearing = (2.5*np.pi - self.dip_direction)%(2*np.pi)
  
		# if self.dip == 0:
		# 	print()

		if self.time == 0 or self.ui == 0 or isinstance(self.ui, (int,float))==False or self.predecessor is None:
	  
			dip_x = np.arctan( np.tan(self.dip) * np.cos(abs(0 - dip_direction_bearing)))
			dip_y = np.arctan( np.tan(self.dip) * np.cos(abs(0.5*np.pi - dip_direction_bearing)))

			if self.time == 0:
				a_mag_x = self.gradients_x + np.cos(dip_x)*self.dhdxi
				a_mag_y = self.gradients_y + np.cos(dip_y)*self.dhdyi
			else:
				a_mag_x = self.gradients_x + np.cos(dip_x)*self.dhdxi*self.k_MD
				a_mag_y = self.gradients_y + np.cos(dip_y)*self.dhdyi*self.k_XMD
	  
			self.travel_direction = np.arctan2(a_mag_y, a_mag_x)   # orientation angle based on gravitaional acceleration direction
			self.cross_travel_direction = self.travel_direction + 0.5*np.pi

		elif self.ui > 0 and self.predecessor is not None:
			self.travel_direction = np.arctan2(self.uy, self.ux)   # orientation angle based on velocity components
			self.cross_travel_direction = self.travel_direction + 0.5*np.pi

		cur_travel_angle_bearing = (2.5*np.pi - self.travel_direction)%(2*np.pi)
		cross_travel_angle_bearing = (2.5*np.pi - self.cross_travel_direction)%(2*np.pi)

		self.dip_travel_direction = np.arctan( np.tan(self.dip) * np.cos(abs(cur_travel_angle_bearing - dip_direction_bearing)))
		self.cross_dip_travel_direction = np.arctan( np.tan(self.dip) * np.cos(abs(cross_travel_angle_bearing - dip_direction_bearing)))

	# compute bed normal stress
	def compute_sigma(self, g):
		self.sigma = self.rho*self.hi*g*np.cos(self.dip_travel_direction)
		if self.sigma <= 0:  # sigma must be zero or positive value
			self.sigma = 0
		return self.sigma

	# compute rheological shear stress
	def compute_tau_r(self, g):	
		self.tau_r = self.sigma*self.fb + self.rho*g*self.ft*(self.ui**2)
		return self.tau_r
	
	# update lateral pressure coefficient k
	def update_local_kx_ky(self, other_part_list):
		
		if self.time == 0 or (self.predecessor is None) == True:
			# assume it is at hydrostatic pressure like Newtonian Fluid
			self.k_MD = 1 
			self.k_XMD = 1
			self.k_MD_max = 1 
			self.k_MD_min = 1
			self.k_XMD_max = 1 
			self.k_XMD_min = 1
		
		elif self.time > 0 and (self.predecessor is None) == False:
			if len(other_part_list) <= 1:
				strain_MD = 0
				strain_XMD = 0

			elif len(other_part_list) > 1:
				## get previous particle distance - S_old
				# previous particle XY 
				pre_other_parts_xy = [(part_c.predecessor.x, part_c.predecessor.y) for part_c in other_part_list ]
				pre_other_parts_xy_rotated = rotate_part_xy_local((self.predecessor.x, self.predecessor.y), pre_other_parts_xy, self.travel_direction)
				pre_other_parts_xy_rotated_array = np.array(pre_other_parts_xy_rotated)

				# compute distance in x-direction and y-direction
				part_dxl_pre_array = pre_other_parts_xy_rotated_array[:,0] - np.array([self.predecessor.x])
				part_dyl_pre_array = pre_other_parts_xy_rotated_array[:,1] - np.array([self.predecessor.y])
				part_dsl_pre_array = np.power( np.power(part_dxl_pre_array,2) + np.power(part_dyl_pre_array,2) , 0.5)

				## get current particle distance - S_cur
				# current particle XY 
				cur_other_parts_xy = [(part_c.x, part_c.y) for part_c in other_part_list ]
				cur_other_parts_xy_rotated = rotate_part_xy_local((self.x, self.y), cur_other_parts_xy, self.travel_direction)
				cur_other_parts_xy_rotated_array = np.array(cur_other_parts_xy_rotated)

				# compute distance in x-direction
				part_dxl_cur_array = cur_other_parts_xy_rotated_array[:,0] - np.array([self.x])
				part_dyl_cur_array = cur_other_parts_xy_rotated_array[:,1] - np.array([self.y])
				part_dsl_cur_array = np.power( np.power(part_dxl_cur_array,2) + np.power(part_dyl_cur_array,2) , 0.5)

				## compute strain in x- and y-direction : strain = (pre_ds-cur_ds)/pre_ds
				part_s_pre_list = part_dsl_pre_array.tolist()
				part_s_cur_list = part_dsl_cur_array.tolist()

				part_dx_cur_list = part_dxl_cur_array.tolist()
				part_dy_cur_list = part_dyl_cur_array.tolist()

				strain_list = []
				cos_2theta_list = []
				for ds_pre, ds_cur, dx_cur, dy_cur in zip(part_s_pre_list, part_s_cur_list, part_dx_cur_list, part_dy_cur_list):
					
					if ds_pre == 0:
						strain_t = 0
					elif ds_pre > 0:
						strain_t = (ds_pre - ds_cur)/ds_pre

					theta_t = np.arctan2(dy_cur, dx_cur)

					strain_list.append(strain_t)
					cos_2theta_list.append(np.cos(2*theta_t))

				# linear interpolation of strain_list vs cos_2theta_list
				local_X_array = np.array(cos_2theta_list)
				local_A = np.vstack([local_X_array, np.ones(len(local_X_array))]).T

				local_Y_array = np.array(strain_list)

				gradient, intercept = np.linalg.lstsq(local_A, local_Y_array, rcond=None)[0]

				# strain between particles
				strain_MD = intercept + gradient
				strain_XMD = intercept - gradient

			# basal friction for k_MD and k_XMD
			if self.sigma > 0:
				eff_basal_resistance = abs(self.tau_r/self.sigma)
			elif self.sigma <= 0:
				eff_basal_resistance = 0

			## max and min kx and ky coefficients
			if eff_basal_resistance <= np.tan(np.radians(self.phi)):
				
				# movment direction max and min
				self.k_MD_max = (2*(1 + np.sqrt(1 - ((np.cos(np.radians(self.phi)))**2)*(1 + eff_basal_resistance**2)))/((np.cos(np.radians(self.phi)))**2)) - 1
				self.k_MD_min = (2*(1 - np.sqrt(1 - ((np.cos(np.radians(self.phi)))**2)*(1 + eff_basal_resistance**2)))/((np.cos(np.radians(self.phi)))**2)) - 1

				## kx
				if strain_MD <= 0:	# expansion
					self.k_MD = self.k_MD_min	# ka
				elif strain_MD > 0:	# compression
					self.k_MD = self.k_MD_max	# kp

				# cross-movment direction max and min	
				self.k_XMD_max = 0.5*(self.k_MD + 1 + np.sqrt( (self.k_MD-1)**2 + 4*(eff_basal_resistance**2) ))
				self.k_XMD_min = 0.5*(self.k_MD + 1 - np.sqrt( (self.k_MD-1)**2 + 4*(eff_basal_resistance**2) ))

				## kyl
				if strain_XMD <= 0:	# expansion
					self.k_XMD = self.k_XMD_min	# ka
				elif strain_XMD > 0:	# compression
					self.k_XMD = self.k_XMD_max	# kp
		
			elif eff_basal_resistance > np.tan(np.radians(self.phi)):

				# movement direction				
				self.k_MD_max = (1 + (np.sin(np.radians(self.phi)))**2)/(np.cos(np.radians(self.phi)))**2
				self.k_MD_min = self.k_MD_max

				self.k_MD = self.k_MD_max

				# cross-movment direction max and min
				self.k_XMD_max = 1/(1 - np.sin(np.radians(self.phi)))
				self.k_XMD_min = 1/(1 + np.sin(np.radians(self.phi)))

				## kyl
				if strain_XMD <= 0:	# expansion
					self.k_XMD = self.k_XMD_min	# ka
				elif strain_XMD > 0:	# compression
					self.k_XMD = self.k_XMD_max	# kp
				
		return (self.k_MD, self.k_XMD)

	# update lateral pressure coefficient k
	def update_local_kx_ky0(self, other_part_list):
		
		if self.time == 0 or (self.predecessor is None) == True:
			# assume it is at hydrostatic pressure like Newtonian Fluid
			self.k_MD = 1 
			self.k_XMD = 1
		
		elif self.time > 0 and (self.predecessor is None) == False:
			if len(other_part_list) <= 1:
				strain_MD = 0
				strain_XMD = 0

			elif len(other_part_list) > 1:
				## get previous particle distance - S_old
				# previous particle XY 
				pre_other_parts_xy = [(part_c.predecessor.x, part_c.predecessor.y) for part_c in other_part_list ]
				pre_other_parts_xy_rotated = rotate_part_xy_local((self.predecessor.x, self.predecessor.y), pre_other_parts_xy, self.travel_direction)
				pre_other_parts_xy_rotated_array = np.array(pre_other_parts_xy_rotated)

				# compute distance in x-direction and y-direction
				part_dxl_pre_array = pre_other_parts_xy_rotated_array[:,0] - np.array([self.predecessor.x])
				part_dyl_pre_array = pre_other_parts_xy_rotated_array[:,1] - np.array([self.predecessor.y])
				part_dsl_pre_array = np.power( np.power(part_dxl_pre_array,2) + np.power(part_dyl_pre_array,2) , 0.5)

				## get current particle distance - S_cur
				# current particle XY 
				cur_other_parts_xy = [(part_c.x, part_c.y) for part_c in other_part_list ]
				cur_other_parts_xy_rotated = rotate_part_xy_local((self.x, self.y), cur_other_parts_xy, self.travel_direction)
				cur_other_parts_xy_rotated_array = np.array(cur_other_parts_xy_rotated)

				# compute distance in x-direction
				part_dxl_cur_array = cur_other_parts_xy_rotated_array[:,0] - np.array([self.x])
				part_dyl_cur_array = cur_other_parts_xy_rotated_array[:,1] - np.array([self.y])
				part_dsl_cur_array = np.power( np.power(part_dxl_cur_array,2) + np.power(part_dyl_cur_array,2) , 0.5)

				## compute strain in x- and y-direction : strain = (pre_ds-cur_ds)/pre_ds
				part_s_pre_list = part_dsl_pre_array.tolist()
				part_s_cur_list = part_dsl_cur_array.tolist()

				part_dx_cur_list = part_dxl_cur_array.tolist()
				part_dy_cur_list = part_dyl_cur_array.tolist()

				strain_list = []
				cos_2theta_list = []
				for ds_pre, ds_cur, dx_cur, dy_cur in zip(part_s_pre_list, part_s_cur_list, part_dx_cur_list, part_dy_cur_list):
					
					if ds_pre == 0:
						strain_t = 0
					elif ds_pre > 0:
						strain_t = (ds_pre - ds_cur)/ds_pre

					theta_t = np.arctan2(dy_cur, dx_cur)

					strain_list.append(strain_t)
					cos_2theta_list.append(np.cos(2*theta_t))

				# linear interpolation of strain_list vs cos_2theta_list
				local_X_array = np.array(cos_2theta_list)
				local_A = np.vstack([local_X_array, np.ones(len(local_X_array))]).T

				local_Y_array = np.array(strain_list)

				gradient, intercept = np.linalg.lstsq(local_A, local_Y_array, rcond=None)[0]

				# strain between particles
				strain_MD = intercept + gradient
				strain_XMD = intercept - gradient

			## max and min kx and ky coefficients
			if abs(self.tau_r/self.sigma) <= np.tan(np.radians(self.phi)):
	   
				# Rankine active and passive k
				# ka = (1 - np.sin(np.radians(self.phi)))/(1 + np.sin(np.radians(self.phi)))
				# kp = 1/ka
				
				# movment direction max and min
				k_MD_max = (2*(1 + np.sqrt(1 - ((np.cos(np.radians(self.phi)))**2)*(1+(self.tau_r/self.sigma)**2)))/((np.cos(np.radians(self.phi)))**2)) - 1
				k_MD_min = (2*(1 - np.sqrt(1 - ((np.cos(np.radians(self.phi)))**2)*(1+(self.tau_r/self.sigma)**2)))/((np.cos(np.radians(self.phi)))**2)) - 1

				## kx
				if strain_MD <= 0:	# expansion
					# self.k_MD = max(self.predecessor.k_MD + strain_MD*(kp-ka)/0.025, k_MD_min)   # ka - incremental decrease or k_MD_min
					self.k_MD = max(self.predecessor.k_MD + strain_MD*200, k_MD_min)   # ka - incremental decrease or k_MD_min
				elif strain_MD > 0:	# compression
					# self.k_MD = min(self.predecessor.k_MD + strain_MD*(kp-ka)/0.05, k_MD_max) 	# kp - incremental increase or k_MD_max
					self.k_MD = min(self.predecessor.k_MD + strain_MD*200, k_MD_max) 	# kp - incremental increase or k_MD_max

				# cross-movment direction max and min	
				k_XMD_max = 0.5*(self.k_MD + 1 + np.sqrt( (self.k_MD-1)**2 + 4*(self.tau_r/self.sigma)**2 ))
				k_XMD_min = 0.5*(self.k_MD + 1 - np.sqrt( (self.k_MD-1)**2 + 4*(self.tau_r/self.sigma)**2 ))

				## kyl
				if strain_XMD <= 0:	# expansion
					# self.k_XMD = max(self.predecessor.k_XMD + strain_XMD*(kp-ka)/0.025, k_XMD_min)   # ka - incremental decrease or k_XMD_min  
					self.k_XMD = max(self.predecessor.k_XMD + strain_XMD*200, k_XMD_min)   # ka - incremental decrease or k_XMD_min  
				elif strain_XMD > 0:	# compression
					# self.k_XMD = min(self.predecessor.k_XMD + strain_XMD*(kp-ka)/0.05, k_XMD_max) 	 # kp - incremental increase or k_XMD_max
					self.k_XMD = min(self.predecessor.k_XMD + strain_XMD*200, k_XMD_max) 	 # kp - incremental increase or k_XMD_max
		
			elif abs(self.tau_r/self.sigma) > np.tan(np.radians(self.phi)):

				# movement direction				
				k_MD_comp = (1 + (np.sin(np.radians(self.phi)))**2)/(np.cos(np.radians(self.phi)))**2
	
				if strain_MD <= 0:	# expansion
					# self.k_MD = max(self.predecessor.k_MD + strain_MD*(kp-ka)/0.025, k_MD_comp)   # ka - incremental decrease or k_MD_min
					self.k_MD = max(self.predecessor.k_MD + strain_MD*200, k_MD_comp)   # ka - incremental decrease or k_MD_min
				elif strain_MD > 0:	# compression
					# self.k_MD = min(self.predecessor.k_MD + strain_MD*(kp-ka)/0.05, k_MD_comp)   # kp - incremental increase or k_MD_max
					self.k_MD = min(self.predecessor.k_MD + strain_MD*200, k_MD_comp)   # kp - incremental increase or k_MD_max

				# cross-movment direction max and min
				k_XMD_max = 1/(1 - np.sin(np.radians(self.phi)))
				k_XMD_min = 1/(1 + np.sin(np.radians(self.phi)))

				## kyl
				if strain_XMD <= 0:	# expansion
					self.k_XMD = k_XMD_min	# ka
				elif strain_XMD > 0:	# compression
					self.k_XMD = k_XMD_max	# kp
	 
				if strain_XMD <= 0:	# expansion
					# self.k_XMD = max(self.predecessor.k_XMD + strain_XMD*(kp-ka)/0.025, k_XMD_min)   # ka - incremental decrease or k_XMD_min  
					self.k_XMD = max(self.predecessor.k_XMD + strain_XMD*200, k_XMD_min)   # ka - incremental decrease or k_XMD_min  
				elif strain_XMD > 0:	# compression
					# self.k_XMD = min(self.predecessor.k_XMD + strain_XMD*(kp-ka)/0.05, k_XMD_max) 	 # kp - incremental increase or k_XMD_max
					self.k_XMD = min(self.predecessor.k_XMD + strain_XMD*200, k_XMD_max) 	 # kp - incremental increase or k_XMD_max
				
		return (self.k_MD, self.k_XMD)

	# compute the new travel direction
	def compute_grad_local(self, g, t_step):
	 
		if self.time == 0:
			
			dip_direction_bearing = (2.5*np.pi - self.dip_direction)%(2*np.pi)
			dip_x = np.arctan( np.tan(self.dip) * np.cos(abs(0 - dip_direction_bearing)))
			dip_y = np.arctan( np.tan(self.dip) * np.cos(abs(0.5*np.pi - dip_direction_bearing)))
	  
			a_x = g*self.gradients_x + g*np.cos(dip_x)*self.dhdxi 	# kx = 1
			a_y = g*self.gradients_y + g*np.cos(dip_y)*self.dhdyi 	# ky = 1

			# applied force in movement direction (MD) and cross-movement directino (XMD)
			self.a_MD = max(a_x, a_y)
			self.a_XMD = min(a_x, a_y)
   
			self.dx_grad_local = self.ux + a_x*t_step
			self.dy_grad_local = self.uy + a_y*t_step

		else:

			# applied force in movement direction (MD) and cross-movement directino (XMD)
			if self.hi > 0:
				self.a_MD = g*np.sin(self.dip_travel_direction) - g*np.cos(self.dip_travel_direction)*self.dh_dMD*self.k_MD -  g*(self.fb*np.cos(self.dip_travel_direction) + (self.ft*(self.ui**2)/self.hi)) - self.Es*(self.ui**2)
			elif self.hi == 0: # no turbulance resistance if depth is zero
				self.a_MD = g*np.sin(self.dip_travel_direction) - g*np.cos(self.dip_travel_direction)*self.dh_dMD*self.k_MD -  g*self.fb*np.cos(self.dip_travel_direction) - self.Es*(self.ui**2)

			self.a_XMD = g*np.sin(self.cross_dip_travel_direction) - g*np.cos(self.dip_travel_direction)*self.dh_dXMD*self.k_XMD

			self.dx_grad_local = self.ux + t_step*(self.a_MD*np.cos(self.travel_direction) + self.a_XMD*np.cos(self.cross_travel_direction))
			self.dy_grad_local = self.uy + t_step*(self.a_MD*np.sin(self.travel_direction) + self.a_XMD*np.sin(self.cross_travel_direction))
   
		return self.dx_grad_local, self.dy_grad_local

	## impact pressure
	def compute_P(self, g):
		if self.ui > 0:
			self.Pi = (self.rho/1000)*(0.5*g*self.hi + (self.ui)**2)  # kPa
		else: # hydrostatic pressure
			self.Pi = (self.rho/1000)*g*self.hi  # kPa
		return self.Pi

	def compute_alpha_P(self, alpha):
		self.Pi = alpha*(self.rho/1000)*((self.ui)**2)  # kPa
		return self.Pi

	def compute_Fr(self, g):
		self.Fr = self.ui/np.sqrt(g*self.hi)
		return self.Fr

	def compute_Fr_P(self, g):
		# https://onlinelibrary.wiley.com/doi/full/10.1002/esp.3744
		# NB alpha = 5.3*(Fr^(-1.5))
		if self.ui > 0:
			self.Fr = self.compute_Fr(g)
			self.Pi = 5.3*(self.Fr**(-1.5))*(self.rho/1000)*((self.ui)**2)  # kPa
		else: # hydrostatic pressure
			self.Pi = (self.rho/1000)*g*self.hi  # kPa
		return self.Pi

	# update time_step
	def update_time(self, d_time=1):
		self.time = self.predecessor.time + d_time

	# replace speed based on COR = coefficient of restitution
	def replace_speed_COR(self, COR):
		self.ui = abs(COR)*self.ui
		self.ux = -COR*self.ux
		self.uy = -COR*self.uy

	def replace_clusterID_and_return_class(self, new_clusterID):
		self.clusterID = new_clusterID
		return self

	## return specific values from the class Particle
	def return_all_param(self):
		return (self.clusterID, self.time, self.si, self.x, self.y, self.z, self.elevation, self.ui, self.ux, self.uy, self.hi, self.Vi, self.Pi)

	def return_all_param_opt(self):
		return (self.clusterID, self.time, self.si, self.x, self.y, self.z, self.ui, self.ux, self.uy, self.Vi)

	def return_everything(self):
		return (self.clusterID, self.time, self.si, self.x, self.y, self.z, self.elevation, self.r, self.materialID, self.phi, self.fb, self.ft, self.rho, self.Es, self.dip, self.dip_direction, self.travel_direction, self.cross_travel_direction, self.gradients_x, self.gradients_y, self.dip_travel_direction, self.cross_dip_travel_direction, self.Vi, self.k_MD, self.k_XMD, self.ll, self.hi, self.div_hi, self.dhdxi, self.dhdyi, self.dh_dMD, self.dh_dXMD, self.ui, self.ux, self.uy, self.a_MD, self.a_XMD, self.sigma, self.tau_r, self.dx_grad_local, self.dy_grad_local, self.Fr, self.Pi)

#################################################################################################################
### child functions
#################################################################################################################

###########################################################################
## child functions - find data from grid
###########################################################################
# rotate XY based on particle location
def rotate_part_xy_local(part_xy, other_part_xy_list, angle_rad):

	offset_x, offset_y = part_xy
	cos_rad = np.cos(angle_rad)
	sin_rad = np.sin(angle_rad)
	
	rotated_xy_list = []
	for x,y in other_part_xy_list:
		adjusted_x = (x - offset_x)
		adjusted_y = (y - offset_y)

		rotated_x = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
		rotated_y = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
		
		rotated_xy_list.append((rotated_x, rotated_y))
	
	return rotated_xy_list

# isolate local cells to compute elevation and path-finding
def local_cell_v3_0(cell_size, x0, y0, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, z_pre):

	# DEM dimension
	dims = [len(gridUniqueY), len(gridUniqueX)]

	# max and min X and Y value
	minX = min(gridUniqueX)
	maxX = max(gridUniqueX)
	minY = min(gridUniqueY)
	maxY = max(gridUniqueY)

	# decimal places
	min_dt_dx = abs(decimal.Decimal(str(deltaX)).as_tuple().exponent)
	min_dt_dy = abs(decimal.Decimal(str(deltaY)).as_tuple().exponent)

	local_xy = []
	local_z = []

	# point is inside the DEM
	if (x0 >= minX) and (x0 <= maxX) and (y0 >= minY) and (y0 <= maxY): 

		try:
			# isolate cell size [n x n] as the local region
			if cell_size > 1:
				# get nearest DEM grid - y - row
				nearest_Y = round( round(y0/deltaY)*deltaY , min_dt_dy)
				row0 = np.where(gridUniqueY == nearest_Y)[0][0]

				# get nearest DEM grid - x - col
				nearest_X = round( round(x0/deltaX)*deltaX , min_dt_dx)
				col0 = np.where(gridUniqueX == nearest_X)[0][0]

				# cell size [n x n]
				cell_list = [ii for ii in range(int((-np.floor(cell_size/2))), int(np.floor(cell_size/2)+1))]

				# find (n x n cell) grids
				for i in cell_list:  
					for j in cell_list:  
						if row0+i<(dims[0]) and row0+i>-1 and col0+j<(dims[1]) and col0+j>-1:
							xt = gridUniqueX[col0+j]
							yt = gridUniqueY[row0+i]
							local_xy.append([xt, yt])

							zt = DEM[row0+i][col0+j]
							local_z.append(zt)

			# take only the elevation from surrounding grid
			elif cell_size == 1:

				# four corners x,y coordinates
				# x
				Cx_min = max(gridUniqueX[0], round( np.floor(x0/deltaX)*deltaX , min_dt_dx))
				Cx_max = min(gridUniqueX[-1], round( np.ceil(x0/deltaX)*deltaX , min_dt_dx))

				# y
				Cy_min = max(gridUniqueY[0], round( np.floor(y0/deltaY)*deltaY , min_dt_dy))
				Cy_max = min(gridUniqueY[-1], round( np.ceil(y0/deltaY)*deltaY , min_dt_dy))

				## reference points
				# exactly at grid points or at the csv file corner 
				if Cx_min == Cx_max and Cy_min == Cy_max:
					row_C1 = np.where(gridUniqueY == Cy_min)[0][0]
					col_C1 = np.where(gridUniqueX == Cx_min)[0][0]
					zt1 = DEM[row_C1][col_C1]
					local_xy.append([Cx_min, Cy_min])
					local_z.append(zt1)

					# add XY points along the axes
					if Cx_min == gridUniqueX[0] and Cy_min == gridUniqueY[0]:  # corner (Xmin, Ymin)
						
						# xmin + deltaX
						local_xy.append([gridUniqueX[1], gridUniqueY[0]])
						local_z.append(DEM[0][1])  # i,j = y,x

						# ymin + deltaY
						local_xy.append([gridUniqueX[0], gridUniqueY[1]])
						local_z.append(DEM[1][0])  # i,j = y,x

					elif Cx_min == gridUniqueX[0] and Cy_min == gridUniqueY[-1]:  # corner (Xmin, Ymax)
						
						# xmin + deltaX
						local_xy.append([gridUniqueX[1], gridUniqueY[-1]])
						local_z.append(DEM[-1][1])  # i,j = y,x

						# ymax - deltaY
						local_xy.append([gridUniqueX[0], gridUniqueY[-2]])
						local_z.append(DEM[-2][0])  # i,j = y,x

					elif Cx_min == gridUniqueX[-1] and Cy_min == gridUniqueY[-1]:  # corner (Xmax, Ymax)
						
						# xmax - deltaX
						local_xy.append([gridUniqueX[-2], gridUniqueY[-1]])
						local_z.append(DEM[-1][-2])  # i,j = y,x

						# ymax - deltaY
						local_xy.append([gridUniqueX[-1], gridUniqueY[-2]])
						local_z.append(DEM[-2][-1])  # i,j = y,x

					elif Cx_min == gridUniqueX[-1] and Cy_min == gridUniqueY[0]:  # corner (Xmax, Ymin)
						
						# xmin + deltaX
						local_xy.append([gridUniqueX[-2], gridUniqueY[0]])
						local_z.append(DEM[0][-2])  # i,j = y,x

						# ymin + deltaY
						local_xy.append([gridUniqueX[-1], gridUniqueY[1]])
						local_z.append(DEM[1][-1])  # i,j = y,x

				# x-coordinate exactly at grid points or at the x_coordinate edge 
				elif Cx_min == Cx_max and Cy_min != Cy_max:

					# Cy_min
					row_C1 = np.where(gridUniqueY == Cy_min)[0][0]
					col_C1 = np.where(gridUniqueX == Cx_min)[0][0]
					zt1 = DEM[row_C1][col_C1]
					local_xy.append([Cx_min, Cy_min])
					local_z.append(zt1)

					# Cy_max
					row_C2 = np.where(gridUniqueY == Cy_max)[0][0]
					col_C2 = np.where(gridUniqueX == Cx_min)[0][0]
					zt2 = DEM[row_C2][col_C2]
					local_xy.append([Cx_min, Cy_max])
					local_z.append(zt2)

					# add XY points along the axes
					if Cx_min == gridUniqueX[0]:  # boundary Xmin
						
						# xmin + deltaX
						local_xy.append([gridUniqueX[1], Cy_min])
						local_z.append(DEM[row_C1][1])  # i,j = y,x

						# xmin + deltaX
						local_xy.append([gridUniqueX[1], Cy_max])
						local_z.append(DEM[row_C2][1])  # i,j = y,x

					elif Cx_min == gridUniqueX[-1]:  # boundary Xmax
						
						# xmax - deltaX
						local_xy.append([gridUniqueX[-2], Cy_min])
						local_z.append(DEM[row_C1][-2])  # i,j = y,x

						# xmax - deltaX
						local_xy.append([gridUniqueX[-2], Cy_max])
						local_z.append(DEM[row_C2][-2])  # i,j = y,x

				# y-coordinate exactly at grid points or at the y_coordinate edge 
				elif Cx_min != Cx_max and Cy_min == Cy_max:

					# Cx_min
					row_C1 = np.where(gridUniqueY == Cy_min)[0][0]
					col_C1 = np.where(gridUniqueX == Cx_min)[0][0]
					zt1 = DEM[row_C1][col_C1]
					local_xy.append([Cx_min, Cy_min])
					local_z.append(zt1)

					# Cx_max
					row_C2 = np.where(gridUniqueY == Cy_min)[0][0]
					col_C2 = np.where(gridUniqueX == Cx_max)[0][0]
					zt2 = DEM[row_C2][col_C2]
					local_xy.append([Cx_max, Cy_min])
					local_z.append(zt2)

					# add XY points along the axes
					if Cy_min == gridUniqueY[0]:  # boundary Ymin
						
						# ymin + deltaY
						local_xy.append([Cx_min, gridUniqueY[1]])
						local_z.append(DEM[1][col_C1])  # i,j = y,x

						# ymax - deltaY
						local_xy.append([Cx_max, gridUniqueY[1]])
						local_z.append(DEM[1][col_C2])  # i,j = y,x

					elif Cy_min == gridUniqueY[-1]:  # boundary Ymax
						
						# ymin + deltaY
						local_xy.append([Cx_min, gridUniqueY[-2]])
						local_z.append(DEM[-2][col_C1])  # i,j = y,x

						# ymax - deltaY
						local_xy.append([Cx_max, gridUniqueY[-2]])
						local_z.append(DEM[-2][col_C2])  # i,j = y,x

				else:

					## C1 - x_min, y_min
					# get nearest DEM grid - y - row
					row_C1 = np.where(gridUniqueY == Cy_min)[0][0]
					col_C1 = np.where(gridUniqueX == Cx_min)[0][0]

					xt1 = gridUniqueX[col_C1]
					yt1 = gridUniqueY[row_C1]
					local_xy.append([xt1, yt1])

					zt1 = DEM[row_C1][col_C1]
					local_z.append(zt1)

					## C2 - x_min, y_max
					# get nearest DEM grid - y - row
					row_C2 = np.where(gridUniqueY == Cy_max)[0][0]
					col_C2 = np.where(gridUniqueX == Cx_min)[0][0]

					xt2 = gridUniqueX[col_C2]
					yt2 = gridUniqueY[row_C2]
					local_xy.append([xt2, yt2])

					zt2 = DEM[row_C2][col_C2]
					local_z.append(zt2)

					## C3 - x_max, y_min
					# get nearest DEM grid - y - row
					row_C3 = np.where(gridUniqueY == Cy_min)[0][0]
					col_C3 = np.where(gridUniqueX == Cx_max)[0][0]

					xt3 = gridUniqueX[col_C3]
					yt3 = gridUniqueY[row_C3]
					local_xy.append([xt3, yt3])

					zt3 = DEM[row_C3][col_C3]
					local_z.append(zt3)

					## C4 - x_max, y_max
					# get nearest DEM grid - y - row
					row_C4 = np.where(gridUniqueY == Cy_max)[0][0]
					col_C4 = np.where(gridUniqueX == Cx_max)[0][0]

					xt4 = gridUniqueX[col_C4]
					yt4 = gridUniqueY[row_C4]
					local_xy.append([xt4, yt4])

					zt4 = DEM[row_C4][col_C4]
					local_z.append(zt4)


		# particle has reached outside the DEM boundary
		# local cell size too big - near the edge
		except Exception as e: 

			# four corners x,y coordinates
			# x
			Cx_min = max(gridUniqueX[0], round( np.floor(x0/deltaX)*deltaX , min_dt_dx))
			Cx_max = min(gridUniqueX[-1], round( np.ceil(x0/deltaX)*deltaX , min_dt_dx))

			# y
			Cy_min = max(gridUniqueY[0], round( np.floor(y0/deltaY)*deltaY , min_dt_dy))
			Cy_max = min(gridUniqueY[-1], round( np.ceil(y0/deltaY)*deltaY , min_dt_dy))

			## reference points
			# exactly at grid points or at the csv file corner 
			if Cx_min == Cx_max and Cy_min == Cy_max:
				row_C1 = np.where(gridUniqueY == Cy_min)[0][0]
				col_C1 = np.where(gridUniqueX == Cx_min)[0][0]
				zt1 = DEM[row_C1][col_C1]
				local_xy.append([Cx_min, Cy_min])
				local_z.append(zt1)

				# add XY points along the axes
				if Cx_min == gridUniqueX[0] and Cy_min == gridUniqueY[0]:  # corner (Xmin, Ymin)
					
					# xmin + deltaX
					local_xy.append([gridUniqueX[1], gridUniqueY[0]])
					local_z.append(DEM[0][1])  # i,j = y,x

					# ymin + deltaY
					local_xy.append([gridUniqueX[0], gridUniqueY[1]])
					local_z.append(DEM[1][0])  # i,j = y,x

				elif Cx_min == gridUniqueX[0] and Cy_min == gridUniqueY[-1]:  # corner (Xmin, Ymax)
					
					# xmin + deltaX
					local_xy.append([gridUniqueX[1], gridUniqueY[-1]])
					local_z.append(DEM[-1][1])  # i,j = y,x

					# ymax - deltaY
					local_xy.append([gridUniqueX[0], gridUniqueY[-2]])
					local_z.append(DEM[-2][0])  # i,j = y,x

				elif Cx_min == gridUniqueX[-1] and Cy_min == gridUniqueY[-1]:  # corner (Xmax, Ymax)
					
					# xmax - deltaX
					local_xy.append([gridUniqueX[-2], gridUniqueY[-1]])
					local_z.append(DEM[-1][-2])  # i,j = y,x

					# ymax - deltaY
					local_xy.append([gridUniqueX[-1], gridUniqueY[-2]])
					local_z.append(DEM[-2][-1])  # i,j = y,x

				elif Cx_min == gridUniqueX[-1] and Cy_min == gridUniqueY[0]:  # corner (Xmax, Ymin)
					
					# xmin + deltaX
					local_xy.append([gridUniqueX[-2], gridUniqueY[0]])
					local_z.append(DEM[0][-2])  # i,j = y,x

					# ymin + deltaY
					local_xy.append([gridUniqueX[-1], gridUniqueY[1]])
					local_z.append(DEM[1][-1])  # i,j = y,x

			# x-coordinate exactly at grid points or at the x_coordinate edge 
			elif Cx_min == Cx_max and Cy_min != Cy_max:

				# Cy_min
				row_C1 = np.where(gridUniqueY == Cy_min)[0][0]
				col_C1 = np.where(gridUniqueX == Cx_min)[0][0]
				zt1 = DEM[row_C1][col_C1]
				local_xy.append([Cx_min, Cy_min])
				local_z.append(zt1)

				# Cy_max
				row_C2 = np.where(gridUniqueY == Cy_max)[0][0]
				col_C2 = np.where(gridUniqueX == Cx_min)[0][0]
				zt2 = DEM[row_C2][col_C2]
				local_xy.append([Cx_min, Cy_max])
				local_z.append(zt2)

				# add XY points along the axes
				if Cx_min == gridUniqueX[0]:  # boundary Xmin
					
					# xmin + deltaX
					local_xy.append([gridUniqueX[1], Cy_min])
					local_z.append(DEM[row_C1][1])  # i,j = y,x

					# xmin + deltaX
					local_xy.append([gridUniqueX[1], Cy_max])
					local_z.append(DEM[row_C2][1])  # i,j = y,x

				elif Cx_min == gridUniqueX[-1]:  # boundary Xmax
					
					# xmax - deltaX
					local_xy.append([gridUniqueX[-2], Cy_min])
					local_z.append(DEM[row_C1][-2])  # i,j = y,x

					# xmax - deltaX
					local_xy.append([gridUniqueX[-2], Cy_max])
					local_z.append(DEM[row_C2][-2])  # i,j = y,x


			# y-coordinate exactly at grid points or at the y_coordinate edge 
			elif Cx_min != Cx_max and Cy_min == Cy_max:

				# Cx_min
				row_C1 = np.where(gridUniqueY == Cy_min)[0][0]
				col_C1 = np.where(gridUniqueX == Cx_min)[0][0]
				zt1 = DEM[row_C1][col_C1]
				local_xy.append([Cx_min, Cy_min])
				local_z.append(zt1)

				# Cx_max
				row_C2 = np.where(gridUniqueY == Cy_min)[0][0]
				col_C2 = np.where(gridUniqueX == Cx_max)[0][0]
				zt2 = DEM[row_C2][col_C2]
				local_xy.append([Cx_max, Cy_min])
				local_z.append(zt2)

				# add XY points along the axes
				if Cy_min == gridUniqueY[0]:  # boundary Ymin
					
					# ymin + deltaY
					local_xy.append([Cx_min, gridUniqueY[1]])
					local_z.append(DEM[1][col_C1])  # i,j = y,x

					# ymax - deltaY
					local_xy.append([Cx_max, gridUniqueY[1]])
					local_z.append(DEM[1][col_C2])  # i,j = y,x

				elif Cy_min == gridUniqueY[-1]:  # boundary Ymax
					
					# ymin + deltaY
					local_xy.append([Cx_min, gridUniqueY[-2]])
					local_z.append(DEM[-2][col_C1])  # i,j = y,x

					# ymax - deltaY
					local_xy.append([Cx_max, gridUniqueY[-2]])
					local_z.append(DEM[-2][col_C2])  # i,j = y,x

			else:

				## C1 - x_min, y_min
				# get nearest DEM grid - y - row
				row_C1 = np.where(gridUniqueY == Cy_min)[0][0]
				col_C1 = np.where(gridUniqueX == Cx_min)[0][0]

				xt1 = gridUniqueX[col_C1]
				yt1 = gridUniqueY[row_C1]
				local_xy.append([xt1, yt1])

				zt1 = DEM[row_C1][col_C1]
				local_z.append(zt1)

				## C2 - x_min, y_max
				# get nearest DEM grid - y - row
				row_C2 = np.where(gridUniqueY == Cy_max)[0][0]
				col_C2 = np.where(gridUniqueX == Cx_min)[0][0]

				xt2 = gridUniqueX[col_C2]
				yt2 = gridUniqueY[row_C2]
				local_xy.append([xt2, yt2])

				zt2 = DEM[row_C2][col_C2]
				local_z.append(zt2)

				## C3 - x_max, y_min
				# get nearest DEM grid - y - row
				row_C3 = np.where(gridUniqueY == Cy_min)[0][0]
				col_C3 = np.where(gridUniqueX == Cx_max)[0][0]

				xt3 = gridUniqueX[col_C3]
				yt3 = gridUniqueY[row_C3]
				local_xy.append([xt3, yt3])

				zt3 = DEM[row_C3][col_C3]
				local_z.append(zt3)

				## C4 - x_max, y_max
				# get nearest DEM grid - y - row
				row_C4 = np.where(gridUniqueY == Cy_max)[0][0]
				col_C4 = np.where(gridUniqueX == Cx_max)[0][0]

				xt4 = gridUniqueX[col_C4]
				yt4 = gridUniqueY[row_C4]
				local_xy.append([xt4, yt4])

				zt4 = DEM[row_C4][col_C4]
				local_z.append(zt4)
			
	# point outside the DEM
	else:
		# return itself with same Z-axis value
		local_xy.append([x0, y0])
		
		if z_pre is None:
			local_z.append(0)
		else:
			local_z.append(z_pre)

	return local_xy, local_z

# compute elevation Z value based on local DEM grid cells
def compute_Z_v3_0(part_xy, local_xy, local_z, interp_method):
	'''
	import relevent python modules for provided interpolation type and method chosen
	
	interp_method 
	when len(local_xy) == 1:
		cornor point

	when len(local_xy) == 2:
		linear interpolation between two points

	when len(local_xy) > 2: 
		3D interpolation method (default = 'lin')
	
		'lin' -> scipy.interpolate.griddata function with linear method

		'cubic' -> scipy.interpolate.griddata function with cubic method
	
		'OK xxx'  -> Ordinary Kriging with semi-variance model of xxx (pykrige)
			e.g. 'OK linear' -> Ordinate Kriging with linear semi-variance model

		'UK xxx'  -> Universal Kriging with semi-variance model of xxx (pykrige)
			e.g. 'UK gaussian' -> Universal Kriging with gaussian semi-variance model
	
			Mathematical model for semi-veriograms (kriging):
			1. linear						
			2. power			
			3. gaussian			
			4. spherical			
			5. exponential
	'''

	# both coordinates exactly at grid points or at the csv file corner 
	if len(local_xy) == 1:
		return float(local_z[0])

	# one of coordinate is exactly at grid points or at the coordinate edge
	elif len(local_xy) == 2:

		dx = abs(local_xy[0][0] - local_xy[1][0])
		dy = abs(local_xy[0][1] - local_xy[1][1])

		# linear interpolation along y-axis
		if dx == 0:
			interpolZ = local_z[0] + ((local_z[1] - local_z[0])/(local_xy[1][1] - local_xy[0][1]))*(part_xy[1] - local_xy[0][1])

		# linear interpolation along x-axis
		elif dy == 0:
			interpolZ = local_z[0] + ((local_z[1] - local_z[0])/(local_xy[1][0] - local_xy[0][0]))*(part_xy[0] - local_xy[0][0])

		return float(interpolZ)

	# other cases - cell_size > 1 or one with all the edges
	else:
		try:
			if interp_method == 'linear': # scipy linear interpolation
				interpolZ = griddata(np.array(local_xy), np.array(local_z), (part_xy[0], part_xy[1]), method='linear')

			elif interp_method == 'cubic': # scipy 2D cubic interpolation
				if len(local_xy) < 16:	# when number of point less than 16 - minimum points required to perform bicubic interpolation
					interpolZ = griddata(np.array(local_xy), np.array(local_z), (part_xy[0], part_xy[1]), method='linear')
				else:		
					interpolZ = griddata(np.array(local_xy), np.array(local_z), (part_xy[0], part_xy[1]), method='cubic')

			elif interp_method[:2] == 'OK':  # ordinary Kriging interpolation

				# check that there DEM is not flat
				if abs(max(local_z) - min(local_z)) < 0.01:
					if np.unique(local_z) == 1: # if only one value, then use the single unique value
						interpolZ = float(np.unique(local_z))
					
					else: # use scipy bilinear interpolation
						interpolZ = griddata(np.array(local_xy), np.array(local_z), (part_xy[0], part_xy[1]), method='linear')

				else:
					interp_method_list = interp_method.split(' ')		
					variogram_model = interp_method_list[1]
					
					inputX, inputY = np.transpose(np.array(local_xy))

					tempInterpolated = OrdinaryKriging(inputX, inputY, np.array(local_z), variogram_model=variogram_model)
					interpolZ, stdZ = tempInterpolated.execute('points', part_xy[0], part_xy[1])
					interpolZ = float(interpolZ)
				
			elif interp_method[:2] == 'UK':  # universal Kriging interpolation

				# check that there DEM is not flat
				if abs(max(local_z) - min(local_z)) < 0.01:
					if np.unique(local_z) == 1: # if only one value, then use the single unique value
						interpolZ = float(np.unique(local_z))
					
					else: # use scipy bilinear interpolation
						interpolZ = griddata(np.array(local_xy), np.array(local_z), (part_xy[0], part_xy[1]), method='linear')

				else:

					interp_method_list = interp_method.split(' ')		
					variogram_model = interp_method_list[1]

					inputX, inputY = np.transpose(np.array(local_xy))

					tempInterpolated = UniversalKriging(inputX, inputY, np.array(local_z), variogram_model=variogram_model)
					interpolZ, stdZ = tempInterpolated.execute('points', part_xy[0], part_xy[1])
					interpolZ = float(interpolZ)

			return float(interpolZ)

		except:
			# print('Error: interpolation method is incorrect or unrecognized - check interp_method input')
			return None

# find location-based material data - 
def local_mat_v2_0(mat_dict, x0, y0, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY):

	# DEM dimension
	dims = [len(gridUniqueY), len(gridUniqueX)]

	# decimal places
	min_dt_dx = abs(decimal.Decimal(str(deltaX)).as_tuple().exponent)
	min_dt_dy = abs(decimal.Decimal(str(deltaY)).as_tuple().exponent)

	try:
		# get nearest DEM grid - y - row
		nearest_Y = round( round(y0/deltaY)*deltaY , min_dt_dy)
		row0 = np.where(gridUniqueY == nearest_Y)[0][0]

		# get nearest DEM grid - x - col
		nearest_X = round( round(x0/deltaX)*deltaX , min_dt_dx)
		col0 = np.where(gridUniqueX == nearest_X)[0][0]

		mat_id = MAT[row0][col0]

	except IndexError:
		most_common_material_mode, most_common_material_count = mode(MAT)
		most_common_material_mode_list = most_common_material_mode.tolist()
		most_common_material_mode_list2 = most_common_material_mode_list[0]
		most_common_material_count_list = most_common_material_count.tolist()
		most_common_material_count_list2 = most_common_material_count_list[0]

		highest_count_idx = most_common_material_count_list2.index(max(most_common_material_count_list2))
		mat_id = most_common_material_mode_list2[highest_count_idx]

	f = mat_dict[mat_id]['f']
	Es = mat_dict[mat_id]['Es']
	density = mat_dict[mat_id]['density']
	phi = mat_dict[mat_id]['phi']

	return f, Es, density, phi

# find materialID for a location
def local_matID_v1_0(x0, y0, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, matID_pre):

	# DEM dimension
	dims = [len(gridUniqueY), len(gridUniqueX)]

	# decimal places
	min_dt_dx = abs(decimal.Decimal(str(deltaX)).as_tuple().exponent)
	min_dt_dy = abs(decimal.Decimal(str(deltaY)).as_tuple().exponent)

	try:
		# get nearest DEM grid - y - row
		nearest_Y = round( round(y0/deltaY)*deltaY , min_dt_dy)
		row0 = np.where(gridUniqueY == nearest_Y)[0][0]

		# get nearest DEM grid - x - col
		nearest_X = round( round(x0/deltaX)*deltaX , min_dt_dx)
		col0 = np.where(gridUniqueX == nearest_X)[0][0]

		mat_id = int(MAT[row0][col0])

	except IndexError:
		mat_id = int(matID_pre)

	return mat_id

# get row and col id (ij coordinate) from x,y-coordinates
def compute_ij(x0, y0, gridUniqueX, gridUniqueY, deltaX, deltaY):

	import decimal
	import numpy as np

	min_dx_dp = abs(decimal.Decimal(str(deltaX)).as_tuple().exponent)
	min_dy_dp = abs(decimal.Decimal(str(deltaY)).as_tuple().exponent)

	# get nearest DEM grid - y - row
	nearest_Y = round( round(y0/deltaY)*deltaY , min_dy_dp)

	if nearest_Y not in gridUniqueY:
		row0 = None
	else:
		row0 = np.where(gridUniqueY == nearest_Y)[0][0]

	# get nearest DEM grid - x - col
	nearest_X = round( round(x0/deltaX)*deltaX , min_dx_dp)
	if nearest_X not in gridUniqueX:
		col0 = None
	else:
		col0 = np.where(gridUniqueX == nearest_X)[0][0]

	return (row0, col0)

###########################################################################
## child functions - setup particles and clusters at time step = 0 
###########################################################################

# find the optimal value of k for k-mean clustering algorithm using the elbow method
def kmeans_findOptK(inputSource, method='elbow', old_k_value=None, max_k_value=10):
	'''
	find optimal k for step 1 - source:
	https://learn.scientificprogramming.io/python-k-means-data-clustering-and-finding-of-the-best-k-485f66297c06

	method options:     'elbow' = elbow method
						'silhouette' = silhouette coefficient method
						'both' = perform both elbow and silhouette coefficient method

	'''

	# import required python libraries 
	# import pandas as pd
	# from sklearn.cluster import KMeans

	# if method in ['silhouette', 'both']:
	# 	from sklearn.metrics import silhouette_score

	# convert input files into panda dataframe format
	A = pd.DataFrame(inputSource)
	#A = pd.read_csv(inputSource)
	
	## compute the sum of squared distances of the samples to their closest cluster center - as the cost of computation 
	kCost_list = []
	for k in range (1, max_k_value+1):
		tempList = [k]
	 
		# Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
		kmeans_model = KMeans(n_clusters=k, random_state=1, n_init=10).fit(A.iloc[:, :])
	 
		# Sum of distances of samples to their closest cluster center
		interia = kmeans_model.inertia_
		tempList.append(interia)

		# calculate decrease percentage of computation cost
		if method in ['elbow', 'both']: 
			if k > 1:
				delta_cost = 100*(kCost_list[k-2][1] - interia)/(kCost_list[k-2][1])
				tempList.append(delta_cost)
			elif k == 1:
				tempList.append(100)

			if method != 'both':
				tempList.append(0)

		# compute the overall average silhouette value
		if method in ['silhouette', 'both']: 
			if k > 1:
				cluster_labels = kmeans_model.fit_predict(A)
				silhouette_avg = silhouette_score(A, cluster_labels)
			else:
				silhouette_avg = 0.0

			if method != 'both':
				tempList.append(0)

			tempList.append(silhouette_avg)

		kCost_list.append(tempList)

	## compute optimal k value
	# find k value at which the decrease percentage is less significant henceforth - elbow method
	if method in ['elbow', 'both']: 

		max_cost_diff = 0   
		max_cost_k = 1

		for loopK in range(len(kCost_list)-1):
			cost_diff_current = abs(kCost_list[loopK][2]-kCost_list[loopK+1][2])    # compute ratio of cost

			# store the index where k value gives the value of max cost ratio
			if loopK == 0:
				max_cost_diff = cost_diff_current

			elif max_cost_diff < cost_diff_current and loopK > 0:
				max_cost_k = kCost_list[loopK][0]
				max_cost_diff = cost_diff_current

			kCost_list[loopK].append(cost_diff_current)     # store cost_diff
		kCost_list[-1].append(0.01)

	# find value of k at which silhouette coefficient is largest - silhouette method
	if method in ['silhouette', 'both']:
		silhouette_avg_k = 0
		max_silhouette_avg = 0
		for loopK in range(len(kCost_list)):
			if max_silhouette_avg < kCost_list[loopK][-2]:
				max_silhouette_avg = kCost_list[loopK][-2]
				silhouette_avg_k = kCost_list[loopK][0]
	

	# additional k mean filtering method
	if old_k_value == None:
		if method == 'elbow':   
			k_value = max_cost_k

		elif method == 'silhouette':    
			k_value = silhouette_avg_k

		elif method == 'both':  
			k_value = max([max_cost_k, silhouette_avg_k])

	else:

		if method == 'elbow':   
			k_value = min(old_k_value, max_cost_k)

		elif method == 'silhouette':    
			k_value = min(old_k_value, silhouette_avg_k)

		elif method == 'both':  
			k_value = min(old_k_value, max([max_cost_k, silhouette_avg_k]))

	return int(k_value)

# use the optimal k value to sort the 
def kmeans_sortPts(inputSource, opt_k_value):

	# import numpy as np
	# import pandas as pd
	# from sklearn.cluster import KMeans

	# convert input files into panda dataframe format
	A = pd.DataFrame(inputSource)

	#inputSource.tolist()

	# if there is more than one cluster sort them
	if opt_k_value > 1:

		## assign each particles to each 
		kmeans_model = KMeans(n_clusters=opt_k_value, random_state=1, n_init=10).fit(A.iloc[:, :])
		k_fit = kmeans_model.fit_predict(A)
		cluster_center_XY = kmeans_model.cluster_centers_
		cluster_center_XY = cluster_center_XY.tolist()

		# add clustering to the source points
		sorted_inputSource_d = {}       # empty list to add point based on cluster
		sorted_inputSource_d_ID = {}    # empty list to add point ID based on cluster
		for loopDict in range(opt_k_value): 
			sorted_inputSource_d[loopDict] = []
			sorted_inputSource_d_ID[loopDict] = []

		for loopA in range(len(inputSource)):

			# sort the points to separate list based on the cluster group number
			clusterNum = int(k_fit[loopA])
			sorted_inputSource_d[clusterNum].append(inputSource[loopA].tolist())
			sorted_inputSource_d_ID[clusterNum].append(loopA)

		sorted_inputSource = []
		sorted_inputSource_ID = []
		for loopList in range(opt_k_value):
			sorted_inputSource.append(sorted_inputSource_d[loopList])
			sorted_inputSource_ID.append(sorted_inputSource_d_ID[loopList])

		return cluster_center_XY, sorted_inputSource, sorted_inputSource_ID

	# if only one cluster, then export central point
	else:

		kmeans_model = KMeans(n_clusters=1, random_state=1, max_iter=500, tol=0.001, n_init=10).fit(A.iloc[:, :])
		#k_fit = kmeans_model.fit_predict(A)
		cluster_center_XY = kmeans_model.cluster_centers_
		cluster_center_XY = cluster_center_XY.tolist()

		return cluster_center_XY[0]

## use for confined fluid - use cell value
# sort out source file into seperate clusters - regular spacing of particles
def sort_part_xyVud_Cell_v4_0(source_file_name, flowpath_file_name, initial_velocity, cell_size, interp_method, part_num_per_cell):

	# load source data
	flowpath_xyz = np.loadtxt(flowpath_file_name, delimiter=',')
	# flowpath_xyz_list = flowpath_xyz.tolist()

	source_xyd = np.loadtxt(source_file_name, delimiter=',')
	source_xyd_list = source_xyd.tolist()

	# get source DEM data
	source_DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(source_file_name, exportAll=True)
	flowpath_DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(flowpath_file_name, exportAll=True)

	# remove source with depth = 0
	sorted_cell_xy_list = []
	sorted_cell_x_list = []
	sorted_cell_y_list = []
	sorted_cell_xyd_list = []
	sorted_cell_xyz_list = []
	for source_xyd_i in source_xyd_list:
		s_x, s_y, s_d = source_xyd_i

		if s_d > 0 and (s_x < gridUniqueX[-1] and s_y < gridUniqueY[-1]):
			# index number of current grid
			s_row_idx, s_col_idx = compute_ij(s_x, s_y, gridUniqueX, gridUniqueY, deltaX, deltaY)  # row = y, col = x
			
			# depth of neighboring grid
			up_grid_d = source_DEM[s_row_idx+1][s_col_idx]	# up-cell (y-direciton)
			right_grid_d = source_DEM[s_row_idx][s_col_idx+1]  # right-cell (x-direciton)

			if up_grid_d == 0 or right_grid_d == 0:
				continue
			elif up_grid_d > 0 and right_grid_d > 0:

				# bilinear interpolation
				cell_d = 0.25*(s_d + up_grid_d + right_grid_d + source_DEM[s_row_idx+1][s_col_idx+1])
				cell_z = 0.25*(flowpath_DEM[s_row_idx][s_col_idx] + flowpath_DEM[s_row_idx+1][s_col_idx] + flowpath_DEM[s_row_idx][s_col_idx+1] + flowpath_DEM[s_row_idx+1][s_col_idx+1])

				sorted_cell_x_list.append(0.5*deltaX + s_x)
				sorted_cell_y_list.append(0.5*deltaY + s_y)
				sorted_cell_xy_list.append((0.5*deltaX + s_x , 0.5*deltaY + s_y))
				sorted_cell_xyd_list.append((0.5*deltaX + s_x , 0.5*deltaY + s_y, cell_d))
				sorted_cell_xyz_list.append((0.5*deltaX + s_x , 0.5*deltaY + s_y, cell_z))

	sorted_source_xyd = np.array(sorted_cell_xyd_list)
	# sorted_source_xyz = np.array(sorted_cell_xyz_list)

	if part_num_per_cell is None:
		part_num_per_cell = 1
  
	# part_xy = sorted_source_xyd[:,[0,1]]
	# tree_new = KDTree(np.array(sorted_cell_xy_list))

	opt_k_value = kmeans_findOptK(sorted_source_xyd[:,[0,1]], method='elbow', old_k_value=None, max_k_value=10)
	opt_k_value = int(opt_k_value)
 
	if opt_k_value > 1:
		cluster_cent_xy, cluster_xy, sorted_cluster_xy_ID = kmeans_sortPts(sorted_source_xyd[:,[0,1]], opt_k_value)
	else:
		sorted_cluster_xy_ID = [[idx for idx in range(len(sorted_cell_xyd_list))]]

	# initial velocity - assume it is equal for all sources
	# u_x
	if initial_velocity[0] != None: 
		ux_0 = initial_velocity[0]
	else:
		ux_0 = 0
	# u_y
	if initial_velocity[1] != None: 
		uy_0 = initial_velocity[1]
	else:
		uy_0 = 0
	# u0 
	u0_mag = np.sqrt(ux_0**2 + uy_0**2) 

	# empty lists to store particle data
	all_part_xy = []
	cluster_V = []
	cluster_u = []
	cluster_u_x = []
	cluster_u_y = []
	cluster_h = []
	cluster_dhdx = []
	cluster_dhdy = []
	for cID in range(opt_k_value):
		all_part_xy.append(deepcopy([]))
		cluster_V.append(deepcopy([]))
		cluster_u.append(deepcopy([]))
		cluster_u_x.append(deepcopy([]))
		cluster_u_y.append(deepcopy([]))
		cluster_h.append(deepcopy([]))
		cluster_dhdx.append(deepcopy([]))
		cluster_dhdy.append(deepcopy([]))

	part_num_per_side = np.floor(np.sqrt(part_num_per_cell))

	particle_radius = 0.5*min((deltaX/part_num_per_side), (deltaY/part_num_per_side))

	# total_cell_num = int((len(sorted_cell_x_list))*(len(sorted_cell_y_list)))
	total_cell_num = int((len(sorted_cell_xy_list)))
	total_part_num = int(total_cell_num*(part_num_per_side**2))
 
	# V0
	V_sum = sum([(deltaX*deltaY)*d for x,y,d in sorted_cell_xyd_list])
	V0 = V_sum/total_part_num
	# V_cell = V_sum/total_cell_num

	if part_num_per_side%2 == 1: # odd
		cell_range = [ii for ii in range(-int(np.floor(part_num_per_side*0.5)), int(np.floor(part_num_per_side*0.5))+1)]
		dist_list = deepcopy(cell_range)

	elif part_num_per_side%2 == 0: # even
		cell_range = [ii for ii in range(-int(np.floor(part_num_per_side*0.5)), int(np.floor(part_num_per_side*0.5))+1) if ii != 0]

		dist_list = []
		for cc in cell_range:
			if cc < 0:
				dist_list.append(cc+0.5)
			elif cc > 0:
				dist_list.append(cc-0.5)

	for pID,(xx,yy,zz) in enumerate(sorted_cell_xyz_list):

		# create particle in cell coordinates
		px_list = [xx + (deltaX/part_num_per_side)*ii for ii in dist_list]
		py_list = [yy + (deltaY/part_num_per_side)*ii for ii in dist_list]

		# determine the cluster ID
		for px in px_list:
			for py in py_list:

				# local DEM
				local_xy_start, local_d_start = local_cell_v3_0(cell_size, px, py, source_DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)

				## get depth 
				dd = compute_Z_v3_0([px, py], local_xy_start, local_d_start, interp_method)

				## get depth gradient	
				# least-squares problem - get local gradient in X- and Y-direction
				d_local = LinearRegression().fit(np.array(local_xy_start), np.array(local_d_start))
				h_gradients = d_local.coef_

				dhdxi = -h_gradients[0]
				dhdyi = -h_gradients[1]
				
				if opt_k_value == 1:
					
					all_part_xy[0].append([px, py])

					cluster_u[0].append(u0_mag)
					cluster_u_x[0].append(ux_0)
					cluster_u_y[0].append(uy_0)
					cluster_V[0].append(V0)

					cluster_h[0].append(dd)
					cluster_dhdx[0].append(dhdxi)
					cluster_dhdy[0].append(dhdyi)
				
				else:
					for cID in range(opt_k_value):
						if pID in sorted_cluster_xy_ID[cID]:

							all_part_xy[cID].append([px,py])

							cluster_u[cID].append(u0_mag)
							cluster_u_x[cID].append(ux_0)
							cluster_u_y[cID].append(uy_0)
							cluster_V[cID].append(V0)

							cluster_h[cID].append(dd)
							cluster_dhdx[cID].append(dhdxi)
							cluster_dhdy[cID].append(dhdyi)

							break

	return all_part_xy, cluster_V, cluster_u, cluster_u_x, cluster_u_y, cluster_h, cluster_dhdx, cluster_dhdy, total_part_num, particle_radius

## use to non-confiend fluid - SPH Gaussian
# sort out source file into seperate clusters - regular spacing of particles
def sort_part_xyVud_SPH_v4_0(source_file_name, flowpath_file_name, initial_velocity, l_dp_min, part_num_per_cell):

	# load source data
	flowpath_xyz = np.loadtxt(flowpath_file_name, delimiter=',')
	flowpath_xyz_list = flowpath_xyz.tolist()

	source_xyd = np.loadtxt(source_file_name, delimiter=',')
	source_xyd_list = source_xyd.tolist()

	# get source DEM data
	source_DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(source_file_name, exportAll=True)
	flowpath_DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(flowpath_file_name, exportAll=True)

	# remove source with depth = 0
	sorted_cell_xy_list = []
	sorted_cell_x_list = []
	sorted_cell_y_list = []
	sorted_cell_xyd_list = []
	sorted_cell_xyz_list = []

	for source_xyd_i in source_xyd_list:
		s_x, s_y, s_d = source_xyd_i

		if s_d > 0 and (s_x < gridUniqueX[-1] and s_y < gridUniqueY[-1]):
			# index number of current grid
			s_row_idx, s_col_idx = compute_ij(s_x, s_y, gridUniqueX, gridUniqueY, deltaX, deltaY)  # row = y, col = x
			
			# depth of neighboring grid
			up_grid_d = source_DEM[s_row_idx+1][s_col_idx]	# up-cell (y-direciton)
			right_grid_d = source_DEM[s_row_idx][s_col_idx+1]  # right-cell (x-direciton)

			if up_grid_d == 0 or right_grid_d == 0:
				continue
			elif up_grid_d > 0 and right_grid_d > 0:

				# bilinear interpolation
				cell_d = 0.25*(s_d + up_grid_d + right_grid_d + source_DEM[s_row_idx+1][s_col_idx+1])
				cell_z = 0.25*(flowpath_DEM[s_row_idx][s_col_idx] + flowpath_DEM[s_row_idx+1][s_col_idx] + flowpath_DEM[s_row_idx][s_col_idx+1] + flowpath_DEM[s_row_idx+1][s_col_idx+1])

				sorted_cell_x_list.append(0.5*deltaX + s_x)
				sorted_cell_y_list.append(0.5*deltaY + s_y)
				sorted_cell_xy_list.append((0.5*deltaX + s_x , 0.5*deltaY + s_y))
				sorted_cell_xyd_list.append((0.5*deltaX + s_x , 0.5*deltaY + s_y, cell_d))
				sorted_cell_xyz_list.append((0.5*deltaX + s_x , 0.5*deltaY + s_y, cell_z))

	sorted_source_xyd = np.array(sorted_cell_xyd_list)
	# sorted_source_xyz = np.array(sorted_cell_xyz_list)

	if part_num_per_cell is None:
		part_num_per_cell = 1
  
	# part_xy = sorted_source_xyd[:,[0,1]]
	tree_new = KDTree(np.array(sorted_cell_xy_list))

	opt_k_value = kmeans_findOptK(sorted_source_xyd[:,[0,1]], method='elbow', old_k_value=None, max_k_value=10)
	opt_k_value = int(opt_k_value)
 
	if opt_k_value > 1:
		cluster_cent_xy, cluster_xy, sorted_cluster_xy_ID = kmeans_sortPts(sorted_source_xyd[:,[0,1]], opt_k_value)
	else:
		sorted_cluster_xy_ID = [[idx for idx in range(len(sorted_cell_xyd_list))]]

	# initial velocity - assume it is equal for all sources
	# u_x
	if initial_velocity[0] != None: 
		ux_0 = initial_velocity[0]
	else:
		ux_0 = 0
	# u_y
	if initial_velocity[1] != None: 
		uy_0 = initial_velocity[1]
	else:
		uy_0 = 0
	# u0 
	u0_mag = np.sqrt(ux_0**2 + uy_0**2) 

	# empty lists to store particle data
	all_part_xy = []
	cluster_V = []
	cluster_u = []
	cluster_u_x = []
	cluster_u_y = []
	cluster_h = []
	cluster_dhdx = []
	cluster_dhdy = []
	for cID in range(opt_k_value):
		all_part_xy.append(deepcopy([]))
		cluster_V.append(deepcopy([]))
		cluster_u.append(deepcopy([]))
		cluster_u_x.append(deepcopy([]))
		cluster_u_y.append(deepcopy([]))
		cluster_h.append(deepcopy([]))
		cluster_dhdx.append(deepcopy([]))
		cluster_dhdy.append(deepcopy([]))

	part_num_per_side = np.floor(np.sqrt(part_num_per_cell))

	particle_radius = 0.5*min((deltaX/part_num_per_side), (deltaY/part_num_per_side))

	# total_cell_num = int((len(sorted_cell_x_list))*(len(sorted_cell_y_list)))
	total_cell_num = int((len(sorted_cell_xy_list)))
	total_part_num = int(total_cell_num*(part_num_per_side**2))
	
	# V0
	V_sum = sum([(deltaX*deltaY)*d for x,y,d in sorted_cell_xyd_list])
	V0 = V_sum/total_part_num
	V_cell = V_sum/total_cell_num

	# smoothing length
	ll = l_dp_min*particle_radius

	if part_num_per_side%2 == 1: # odd
		cell_range = [ii for ii in range(-int(np.floor(part_num_per_side*0.5)), int(np.floor(part_num_per_side*0.5))+1)]
		dist_list = deepcopy(cell_range)

	elif part_num_per_side%2 == 0: # even
		cell_range = [ii for ii in range(-int(np.floor(part_num_per_side*0.5)), int(np.floor(part_num_per_side*0.5))+1) if ii != 0]

		dist_list = []
		for cc in cell_range:
			if cc < 0:
				dist_list.append(cc+0.5)
			elif cc > 0:
				dist_list.append(cc-0.5)

	for pID,(xx,yy,zz) in enumerate(sorted_cell_xyz_list):

		# create particle in cell coordinates
		px_list = [xx + (deltaX/part_num_per_side)*ii for ii in dist_list]
		py_list = [yy + (deltaY/part_num_per_side)*ii for ii in dist_list]
  
		# determine the cluster ID
		for px in px_list:
			for py in py_list:

				## get depth and depth gradient
				# find neighbouring particles
				near_part_id_list = sorted(tree_new.query_ball_point((px,py), 2*ll))

				if len(near_part_id_list) == 1:
					near_s_array = np.array([0.0])
					near_dx_array = np.array([0.0])
					near_dy_array = np.array([0.0])
					near_V_array = np.array([V_cell])

				else:
					near_part_xyz_list = [(sorted_cell_xyz_list[n2][0], sorted_cell_xyz_list[n2][1], sorted_cell_xyz_list[n2][2]) for n2 in near_part_id_list]
					near_s_array = np.array([np.sqrt((px - x)**2 + (py - y)**2) for x,y,z in near_part_xyz_list])
					near_dx_array = np.array([(px - x) for x,y,z in near_part_xyz_list])
					near_dy_array = np.array([(py - y) for x,y,z in near_part_xyz_list])
					near_V_array = V_cell*np.ones(len(near_part_id_list))
	
				## compute particle kernel - Gaussian
				a_g = 1/(np.pi*(ll**2))    # 2D
				qq_array = near_s_array/ll
				
				W_array_t = np.absolute(a_g*np.exp(-1*np.power(qq_array, 2)))
				W_array = np.where(near_s_array>(2*ll), 0.0, W_array_t)

				dW_array_t = np.absolute(a_g*2*qq_array*np.exp(-1*np.power(qq_array, 2)))
				dW_array = np.where(near_s_array>(2*ll), 0.0, dW_array_t)
	
				# compute cluster depth
				np.seterr(invalid='ignore')
				dd = np.nansum(W_array*near_V_array)

				# compute cluster depth gradient
				np.seterr(invalid='ignore')
				r_ab_mag_array = np.power(np.power(near_dx_array, 2) + np.power(near_dy_array, 2), 0.5)
				dhdxi = np.nansum(near_V_array*(dW_array/ll)*(near_dx_array/r_ab_mag_array))
				dhdyi = np.nansum(near_V_array*(dW_array/ll)*(near_dy_array/r_ab_mag_array))
				
				if opt_k_value == 1:
					
					all_part_xy[0].append([px, py])

					cluster_u[0].append(u0_mag)
					cluster_u_x[0].append(ux_0)
					cluster_u_y[0].append(uy_0)
					cluster_V[0].append(V0)

					cluster_h[0].append(dd)
					cluster_dhdx[0].append(dhdxi)
					cluster_dhdy[0].append(dhdyi)
				
				else:
					for cID in range(opt_k_value):
						if pID in sorted_cluster_xy_ID[cID]:

							all_part_xy[cID].append([px,py])

							cluster_u[cID].append(u0_mag)
							cluster_u_x[cID].append(ux_0)
							cluster_u_y[cID].append(uy_0)
							cluster_V[cID].append(V0)

							cluster_h[cID].append(dd)
							cluster_dhdx[cID].append(dhdxi)
							cluster_dhdy[cID].append(dhdyi)

							break

	return all_part_xy, cluster_V, cluster_u, cluster_u_x, cluster_u_y, cluster_h, cluster_dhdx, cluster_dhdy, total_part_num, particle_radius

# initial setup for the debris-flow particles and clusters 
def time0_setup_t_v12_0(source_file_name, flowpath_file_name, part_num_per_cell, cell_size, DEM, MAT, material, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, road_xy_list, initial_velocity, g, wall_info, cluster_boundary, cluster_boundary_iter_dict, l_dp_min, t_step, DP=2, particle_only=False, initial_SPH=True):
	
	# subdivide the source region into equally spaced 
	if initial_SPH:   # use Gassian as initial setting 
		start_xy_list, V0_list, u0_list, ux0_list, uy0_list, h0_list, dhdx0_list, dhdy0_list, total_part_num, particle_radius = sort_part_xyVud_SPH_v4_0(source_file_name, flowpath_file_name, initial_velocity, l_dp_min, part_num_per_cell)

	else: 		 # use DEM cell number
		start_xy_list, V0_list, u0_list, ux0_list, uy0_list, h0_list, dhdx0_list, dhdy0_list, total_part_num, particle_radius = sort_part_xyVud_Cell_v4_0(source_file_name, flowpath_file_name, initial_velocity, cell_size, interp_method, part_num_per_cell)

	# print(f"total particle number {total_part_num} with PART_NUM_IN_CELL {part_num_per_cell}")

	cluster_list = []            # flowpath data for each cluster
	clusterID_flow_list = []    # currently active debris-flow cluster ID
	max_cID = 0
	all_part_list = []          # store all particles throughout the analysis

	cID_part_list = []

	## compute average distance between the particles using kdtree - kNN
	# if no wall present, for optimal barrier location selection
	# also compute cluster information
	if wall_info == None and particle_only == False:

		# sort initial set-up into clusters
		for cID, start_xy_l in enumerate(start_xy_list):

			# setup individual particles for each cluster
			cID_part_list_temp = []
			for pID, start_xy in enumerate(start_xy_l):

				# local xy and z cell grid - compute elevation Z based on local grid cell
				local_xy_start, local_z_start = local_cell_v3_0(cell_size, start_xy[0], start_xy[1], DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
				start_part_z = compute_Z_v3_0(start_xy, local_xy_start, local_z_start, interp_method)

				# extract material properties
				materialID = local_matID_v1_0(start_xy[0], start_xy[1], MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
				# f, Es, density, phi = local_mat_v2_0(material, start_xy[0], start_xy[1], MAT, gridUniqueX, gridUniqueY, deltaX, deltaY)

				# create the particle class
				# X, Y, Z, Vi, ui, hi, radius, clusterID, materialID, predecessor
				start_part = Particle(start_xy[0], start_xy[1], start_part_z, V0_list[cID][pID], u0_list[cID][pID], h0_list[cID][pID], particle_radius, cID, materialID, None)

				# add material components
				start_part.insert_material(material)

				# input x- and y-velocity component
				start_part.ux = ux0_list[cID][pID]
				start_part.uy = uy0_list[cID][pID]

				# depth gradient
				start_part.dhdxi = dhdx0_list[cID][pID]
				start_part.dhdyi = dhdy0_list[cID][pID]

				# compute particle bed-normal stress (sigma)
				start_part.compute_dip_and_angles(local_xy_start, local_z_start)
				start_part.compute_sigma(g)

				# compute rheological shear 
				start_part.compute_tau_r(g)

				# compute pressure
				start_part.compute_Fr_P(g)

				start_part.compute_grad_local(g, t_step)
				
				cID_part_list_temp.append(start_part)
				all_part_list.append(start_part)

			cID_part_list.append(cID_part_list_temp)

			# maximum possible source cluster ID
			max_cID = max(max_cID, cID)

		for cID, cID_part_i in enumerate(cID_part_list):

			## clsuter -  clusterID, part_list, predecessor
			cluster_group = Cluster(cID, cID_part_i, None)

			# compute cluster centroid coordinates - xyz
			start_cent_xy = cluster_group.compute_centroid()
			# start_cent_xy = cluster_group.compute_median_centroid()
			# start_cent_xy = cluster_group.compute_max_pressure_centroid()

			local_xy_start, local_z_start = local_cell_v3_0(cell_size, start_cent_xy[0], start_cent_xy[1], DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
			start_cent_z = compute_Z_v3_0(start_cent_xy, local_xy_start, local_z_start, interp_method)
			cluster_group.zc = start_cent_z

			# compute cluster volume
			cluster_group.compute_V()

			cluster_group.compute_av_h_u()	# average depth and velocity

			# compute distance-from-road (D)
			cluster_group.compute_D(road_xy_list)

			# compute pressure
			cluster_group.compute_Fr_P(g)

			## compute external boundary
			# get XY coordinates of the parts contained in the next_cluster
			cluster_group_part_xy_list = [(part.x, part.y) for part in cluster_group.particles]
			
			if cluster_boundary == 'ConcaveHull':	
				# find optimal alpha value and boundary polygon for concave hull using alphashape python library
				cluster_boundary_buffer = particle_radius*l_dp_min   # same as minimum smoothing length

				cluster_CVH_polygon, cluster_inlier_pt = determine_optimal_CVH_boundary(cluster_group_part_xy_list, cluster_boundary_buffer=cluster_boundary_buffer, output_inlier_points=True)
				cluster_group_opt_alpha = determine_optimal_alpha(cluster_inlier_pt, max_alpha=cluster_boundary_iter_dict["max alpha"], min_alpha=cluster_boundary_iter_dict["min alpha"], iter_max=cluster_boundary_iter_dict["max iteration"], dp_accuracy=2)
				# cluster_group_opt_alpha = alphashape.optimizealpha(cluster_inlier_pt, max_iterations=cluster_boundary_iter_dict["max iteration"], lower=cluster_boundary_iter_dict["min alpha"], upper=cluster_boundary_iter_dict["max alpha"], silent=True)

				cluster_boundary_polygon_t = alphashape.alphashape(cluster_inlier_pt, cluster_group_opt_alpha)
				cluster_group_boundary_polygon = cluster_boundary_polygon_t.buffer(cluster_boundary_buffer, cap_style=1, join_style=3)

				cluster_group.concave_hull_alpha = cluster_group_opt_alpha
				cluster_group.boundary_polygon = cluster_group_boundary_polygon

				# boundary XY coordinates and index relative to the input XY coordinate list
				cluster_group_ext_pts_XY = cluster_group_boundary_polygon.exterior.coords.xy
				cluster_group.boundary_pt_xy = cluster_group_ext_pts_XY

			elif cluster_boundary == 'ConvexHull':
				
				# alpha = 0 for Convex hull
				cluster_group.concave_hull_alpha = 0

				# convert corner points to shapely polygon
				cluster_boundary_buffer = particle_radius*l_dp_min   # same as minimum smoothing length
				cluster_group_boundary_polygon = determine_optimal_CVH_boundary(cluster_group_part_xy_list, cluster_boundary_buffer=cluster_boundary_buffer)
				cluster_group.boundary_polygon = cluster_group_boundary_polygon

				# boundary XY coordinates and index relative to the input XY coordinate list
				cluster_group_ext_pts_XY = cluster_group_boundary_polygon.exterior.coords.xy
				cluster_group.boundary_pt_xy = cluster_group_ext_pts_XY
				
			# compute cluster area
			cluster_group.compute_area()

			# add to analysis data
			cluster_list.append([cluster_group])
			clusterID_flow_list.append(cID)

		return cluster_list, clusterID_flow_list, max_cID, [all_part_list], total_part_num, particle_radius

	# for optimal barrier location selection
	elif wall_info is not None or particle_only == True:

		# sort initial set-up into clusters
		for cID, start_xy_l in enumerate(start_xy_list):

			# setup individual particles for each cluster
			for pID, start_xy in enumerate(start_xy_l):

				# local xy and z cell grid - compute elevation Z based on local grid cell
				local_xy_start, local_z_start = local_cell_v3_0(cell_size, start_xy[0], start_xy[1], DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
				start_part_z = compute_Z_v3_0(start_xy, local_xy_start, local_z_start, interp_method)

				# extract material properties
				materialID = local_matID_v1_0(start_xy[0], start_xy[1], MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
				# f, Es, density, phi = local_mat_v2_0(material, start_xy[0], start_xy[1], MAT, gridUniqueX, gridUniqueY, deltaX, deltaY)

				# create the particle class
				# X, Y, Z, Vi, ui, hi, radius, clusterID, materialID, predecessor
				start_part = Particle(start_xy[0], start_xy[1], start_part_z, V0_list[cID][pID], u0_list[cID][pID], h0_list[cID][pID], particle_radius, cID, materialID, None)

				# add material components
				start_part.insert_material(material)

				# input x- and y-velocity component
				start_part.ux = ux0_list[cID][pID]
				start_part.uy = uy0_list[cID][pID]

				# depth gradient
				start_part.dhdxi = dhdx0_list[cID][pID]
				start_part.dhdyi = dhdy0_list[cID][pID]

				# compute particle bed-normal stress (sigma)
				start_part.compute_dip_and_angles(local_xy_start, local_z_start)
				start_part.compute_sigma(g)

				# compute rheological shear and effective viscosity
				start_part.compute_tau_r(g)

				# compute pressure
				start_part.compute_Fr_P(g)

				start_part.compute_grad_local(g, t_step)
				
				all_part_list.append(start_part)

		return [all_part_list], total_part_num, particle_radius


###########################################################################
## child functions - SPH computation
###########################################################################
# multiprocessing depth computation - adaptive smoothing length 
def computing_SPH_p1_MP_v4_0(mp_input_sph_p1):

	n1, cur_part, cur_part_ij, ERODE, g, part_tree_new, ghost_tree_new, part_xy_new, ll, wall_dict, wall_bound_region, all_part_list, all_ghost_list, material = mp_input_sph_p1
 
	try: 

		##############################################
		## find the nerighbor fluid particles for SPH
		##############################################
		# perform KDTree to find particles that are within a (2*smoothing length) from particle position
		near_part_id_list = sorted(part_tree_new.query_ball_point(part_xy_new[n1], 2*ll))

		# exclude particle in the other side of barrier
		if not (wall_dict is None or wall_bound_region is None):

			# check if the SPH interpolation region overlaps with any wall boundary
			SPH_region = Point(part_xy_new[n1][0], part_xy_new[n1][1]).buffer(2*ll)

			intersecting_wall_region_wall_id = []				# wall_id
			intersecting_wall_region_dist_min = []				# distance from region to region
			intersecting_wall_region_dist_min_region = []		# 0 = before region, 1 = after region

			for wall_id in wall_dict.keys():
				wall_multi_poly = wall_dict[wall_id][3]

				if wall_multi_poly.intersects(SPH_region):
					intersecting_wall_region_wall_id.append(wall_id)

					## distance between particle with 
					# before region centroid
					before_region = wall_bound_region[wall_id][0] 
					before_centroid = list(before_region.centroid.xy)
					dist_before_centroid = np.sqrt((before_centroid[0] - part_xy_new[n1][0])**2 + (before_centroid[1] - part_xy_new[n1][1])**2)

					# before region centroid
					after_region = wall_bound_region[wall_id][1] 
					after_centroid = list(after_region.centroid.xy)
					dist_after_centroid = np.sqrt((after_centroid[0] - part_xy_new[n1][0])**2 + (after_centroid[1] - part_xy_new[n1][1])**2)

					# minimum distance
					min_dist_region = min(dist_before_centroid, dist_after_centroid)
					intersecting_wall_region_dist_min.append(min_dist_region)

					if dist_before_centroid >= dist_after_centroid:
						intersecting_wall_region_dist_min_region.append(0)
					else:
						intersecting_wall_region_dist_min_region.append(1)

			# if no barrier is nearby, skip the exclusion process
			if len(intersecting_wall_region_wall_id) > 1:

				# note that is at least one wall region intersects with the SPH region, then there must be at least 2 data in the list
				
				# find two closest regions
				sorted_intersecting_wall_id = [wid for _, wid in sorted(zip(intersecting_wall_region_dist_min, intersecting_wall_region_wall_id))]
				sorted_intersecting_region = [reg for _, reg in sorted(zip(intersecting_wall_region_dist_min, intersecting_wall_region_dist_min_region))]
		
				min_dist_id1, min_dist_id2 = sorted_intersecting_wall_id[0], sorted_intersecting_wall_id[1]
				min_dist_reg1, min_dist_reg2 = sorted_intersecting_region[0], sorted_intersecting_region[1]

				# wall_bound_region = {}		# key = wall_id, value = [shapely polygon before, shapely polygon after]

				## find regions that can be included
				excluding_region = []

				# if only one barrier is nearby or only one barrier was modelled
				if min_dist_id1 == min_dist_id2 and min_dist_reg1 != min_dist_reg2:
					# eg) min_dist_id1 = min_dist_id2 = 1 and min_dist_reg1 = 0(before) and min_dist_reg2 = 1(after)
					# include only nearest region from the particle position

					excluding_region.append(wall_bound_region[min_dist_id1][min_dist_reg2])

				# multiple walls
				elif min_dist_id1 != min_dist_id2:

						if min_dist_reg1 != min_dist_reg2:  
							# two different walls, with wall A in before and wall B in after region - sandwiched between two walls
							# exclude all region except these regions between these two walls

							included_reg1 = wall_bound_region[min_dist_id1][min_dist_reg1]
							included_reg2 = wall_bound_region[min_dist_id2][min_dist_reg2]
							including_region = included_reg1.intersection(included_reg2)		# region between two walls

							for wall_id,nearest_reg in zip(sorted_intersecting_wall_id, sorted_intersecting_region):
								if wall_id == min_dist_id1 or wall_id == min_dist_id2:	# 1st and 2nd nearest wall
									# excluding the further away region for nearest two walls
									excluding_region.append(wall_bound_region[wall_id][int((nearest_reg+1)%2)])

								else:
									# exclude all regions from other walls, while including the including region
									# (other wall region - including region)
									wall_reg1 = wall_bound_region[wall_id][0]
									ex_wall_reg1 = wall_reg1.difference(including_region)
									excluding_region.append(ex_wall_reg1)

									wall_reg2 = wall_bound_region[wall_id][1]
									ex_wall_reg2 = wall_reg2.difference(including_region)
									excluding_region.append(ex_wall_reg2)

						else:  
							# two different walls, with wall A in before and wall B in before region
							# exclude all region except nearest region from nearest wall

							including_region = wall_bound_region[min_dist_id1][min_dist_reg1]		# region closest to point 

							for wall_id,nearest_reg in zip(sorted_intersecting_wall_id, sorted_intersecting_region):
								if wall_id == min_dist_id1:	# nearest wall
									# excluding the further away region for nearest two walls
									excluding_region.append(wall_bound_region[wall_id][int((nearest_reg+1)%2)])
								
								else:
									# exclude all regions from other walls, while including the including region
									# (other wall region - including region)
									wall_reg1 = wall_bound_region[wall_id][0]
									ex_wall_reg1 = wall_reg1.difference(including_region)
									excluding_region.append(ex_wall_reg1)

									wall_reg2 = wall_bound_region[wall_id][1]
									ex_wall_reg2 = wall_reg2.difference(including_region)
									excluding_region.append(ex_wall_reg2)
				

				## exclude particle in excluding regions 
				new_filtered_near_part_id_list = []
				for n2 in near_part_id_list:
					# always add the its own particle position
					if n2 == n1:
						new_filtered_near_part_id_list.append(n2)
					
					# remove points in excluding regions
					else:
						in_excluded_region = False
						for exclude_poly in excluding_region:
							# if any particle is found in the excluding regions, then stop iteration and change boolean
							if Point(part_xy_new[n2][0], part_xy_new[n2][1]).within(exclude_poly):
								in_excluded_region = True
								break							
						
						if in_excluded_region == False:  # if no particle is in any excluded regions
							new_filtered_near_part_id_list.append(n2)

				near_part_id_list = deepcopy(new_filtered_near_part_id_list)
				del new_filtered_near_part_id_list

				
	
		##############################################
		## find the nerighbor ghost particles for SPH
		##############################################
		# perform KDTree to find particles that are within a (2*smoothing length) from particle position
		near_ghost_id_list = sorted(ghost_tree_new.query_ball_point(part_xy_new[n1], 2*ll)) 

		##############################################
		## compute SPH data
		##############################################
		# add fluid particle
		# if none near-by, assume all expansion happening
		if len(near_part_id_list) == 1:  
			near_part_k_list = [cur_part]
			near_part_ghost_list = [cur_part]
		elif len(near_part_id_list) > 1:
			near_part_k_list = [all_part_list[n2] for n2 in near_part_id_list]
			near_part_ghost_list = [all_part_list[n2] for n2 in near_part_id_list]
	
		# add ghost particle
		if len(near_ghost_id_list) > 1: 
			for gn1 in near_ghost_id_list:
				near_part_ghost_list.append(all_ghost_list[gn1])

		# depth, depth gradient, and erosion depth check
		if cur_part_ij[0] is not None and cur_part_ij[1] is not None:
			erode_DEM_ij = float(ERODE[cur_part_ij[0], cur_part_ij[1]])
		else:
			erode_DEM_ij = 0
	
		erode_DEM_ij_new = cur_part.compute_h_local_dh_check_Es_ghost(near_part_ghost_list, material, erode_DEM_ij, ll)    # compute depth, erosion depth, and depth gradient of current particle
	
		# compute stresses (bed-normal stress and shear rheological stress)
		cur_part.compute_sigma(g)
		cur_part.compute_tau_r(g)

		# local lateral pressure coefficient
		cur_part.update_local_kx_ky(near_part_k_list)		# compute lateral pressure coefficient - kx and ky

		# compute apparent surface elevation of debris-flow particles
		cur_part.elevation = cur_part.z + cur_part.hi

		return (cur_part, erode_DEM_ij_new)

	except Exception as e:
	
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

		print("error cuased by: ")
		print(exc_type, fname, exc_tb.tb_lineno)
		print("terminated early")
		print("error cuased by: "+str(e))
		print()

		return None

# computing new depth - adaptive smoothing length 
def compute_SPH_v4_0(successor_part_list, ghost_part_list, ERODE, material, g, l_dp_min, wall_dict, wall_bound_region, gridUniqueX, gridUniqueY, deltaX, deltaY, max_cpu_num):

	## particle data
	# new particle positions
	part_xy_new = np.array([[part_i.x, part_i.y] for part_i in successor_part_list])
	ghost_xy = np.array([[ghost_i.x, ghost_i.y] for ghost_i in ghost_part_list])
	
	# find cell row and column number (ij)
	part_ij_new = [compute_ij(part_i.x, part_i.y, gridUniqueX, gridUniqueY, deltaX, deltaY) for part_i in successor_part_list]
	
	# create KDTree to find nearby particles
	part_tree_new = KDTree(part_xy_new)
 
	# create KDTree to find nearby ghost particles
	ghost_tree_new = KDTree(ghost_xy)

	## adaptive smoothing length from DAN3D
	# other particle density
	min_part_radius = min([part_i.r for part_i in successor_part_list])
 
	# particle old depth
	part_h_old = np.array([part_i.predecessor.hi for part_i in successor_part_list])

	# particle volume
	part_V_old = np.array([part_i.predecessor.Vi for part_i in successor_part_list])
 
	# new smoothing length
	B = 4
	av_h_V = np.mean(part_h_old/part_V_old)
	ll_temp = B/np.sqrt(av_h_V)

	l_dp_max = 10*l_dp_min

	if ll_temp <= l_dp_min*min_part_radius:
		ll = l_dp_min*min_part_radius
	elif ll_temp >= l_dp_max*min_part_radius:
		ll = l_dp_max*min_part_radius
	else:
		ll = ll_temp

	# mp_input_sph_p1  
	mp_input_sph_p1 = [(n1, successor_part_list[n1], part_ij_new[n1], ERODE, g, part_tree_new, ghost_tree_new, part_xy_new, ll, wall_dict, wall_bound_region, successor_part_list, ghost_part_list, material) for n1 in range(len(successor_part_list))]
 
	## multiprocessing set-up
	if mp.cpu_count() >= max_cpu_num:
		cpu_num = max_cpu_num 
	else:
		cpu_num = mp.cpu_count()
	pool_SPH = mp.Pool(cpu_num)

	# compute: depth, erosion depth, depth gradient, stresses, effective viscosity, kx and ky
	successor_part_sph_erode_list = pool_SPH.map(computing_SPH_p1_MP_v4_0, mp_input_sph_p1)
	successor_part_sph_list = [ite[0] for ite in successor_part_sph_erode_list]
	successor_part_erode_depth_list = [ite[1] for ite in successor_part_sph_erode_list]
	# part, n1, near_part_id_list, ll, near_u_array, near_V_array, near_s_array, near_dx_array, near_dy_array
	
	## stop multiprocessing
	pool_SPH.close()
	pool_SPH.join()
 
	# new erode
	ERODE_new = np.zeros(ERODE.shape)
	for (i,j),new_erode_depth in zip(part_ij_new, successor_part_erode_depth_list):
		ERODE_new[i,j] = new_erode_depth

	return successor_part_sph_list, ERODE_new


###########################################################################
## child functions - cluster merge
###########################################################################

# determine optimal alpha to use for alpha_shape function to determine the debris cluster boundary
def determine_optimal_alpha_boundary(points, increment=0.01, max_alpha=0.5, angle_lim=60, tol_angle=10, tol_percent=5, cluster_boundary_buffer=0, dp=2): 
	'''
	## determin the optimal alpha value for alpha_shape function
	criteria for determining optimal alpha value (not confirmed but still operational until proven wrong)
	1. The % change of difference in perimeter and area are both nearly equal to zero (0)
			i.e. delta(area%) = 0 and delta(perimeter%) = 0
	2. The boundary should include all the data point. However, if a encompassing data point is considered to be an outlier, 
		then non-inclusion is acceptable. Outlier data point is when the average distance to another data points are 
		1.5 times greater than the average distance between data points that does not include the checking data point
			i.e. if av(d_i)>1.5*av(d_(all other points except the point i)), when point i is checking point
	3. The alpha value does not cause the splitting of data points into two polygons
			i.e. number of polygon generated == 1

	try to find the minimum alpha value that fits the following function
	'''
	
	# remove repeating points
	points = list(set(points))

	## check if points would create a line or single point -> zero cluster area
	points_array = np.array(points)
	if len(points) <= 1 or len(np.unique(points_array[:, 0])) == 1 or len(np.unique(points_array[:,1])) == 1:
		cent_x, cent_y = np.mean(points_array, axis=0)
		return Point(cent_x, cent_y)
		
	####################################################################################################
	### find outlier points
	####################################################################################################

	##################################################
	## find outline locations - based LocalOutliearFactor (LOF) at initial stage
	##################################################
	# LOF
	raw_points_array = np.array(points)
	debris_lof = LocalOutlierFactor(n_neighbors=20)
	outlier_pred = debris_lof.fit_predict(raw_points_array)	# 1 = inlier, -1 = outliers
	outlier_pred_list = outlier_pred.tolist()

	# find all particle positions not considered as an outlier
	inlier_part_xy = [tuple(points[idx]) for idx,pred in enumerate(outlier_pred_list) if pred == 1]
	inlier_part_xy_array = np.array(inlier_part_xy)

	# find maximum neighbour distance among the inlier points
	nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(inlier_part_xy_array)
	distances, indices = nbrs.kneighbors(inlier_part_xy_array)

	# exclude the distance at 1st index, because distance at 1st index == 0 computing average distance from a particle position
	max_1NN_dist = max([loopDist[1] for loopDist in distances])

	# find all particle positions considered as an outlier
	outlier_part_xy = [tuple(points[idx]) for idx,pred in enumerate(outlier_pred_list) if pred == -1]
	outlier_part_xy_array = np.array(outlier_part_xy)
	
	##################################################
	## refine LOF results
	##################################################
	alpha = 0	# alpha values to consider, start from 0 (i.e. convex hull)
	alpha_list = []
	idx = 0
	opt_alpha = None
	opt_alpha_list = []

	# iteratively change alpha value to find all the inlier points that LOF misplaced
	while alpha <= max_alpha:	

		# perform alpha_shape function with given alpha
		bound_poly = alphashape.alphashape(inlier_part_xy_array, alpha)

		# ensure that only one polygon is generated
		try:
			ext_pts_XY = list(bound_poly.exterior.coords)
			if len(ext_pts_XY) == 0:
				alpha += increment
				continue
		except:
			alpha += increment
			continue

		if alpha == 0: 
			# if concave hull, first store the value and start next iteration
			# store data
			alpha_list.append([alpha, bound_poly.area, bound_poly.length])  # alpha, area, perimeter
			alpha += increment
			idx += 1

		elif alpha > 0:

			# compute perimeter and area
			area_cur = bound_poly.area
			perimeter_cur = bound_poly.length

			# extract area and perimeter from previous iteration
			area_pre = alpha_list[idx-1][1]
			perimeter_pre = alpha_list[idx-1][2]

			# extract area and perimeter from concex hull (alpha = 0)
			area_ch = alpha_list[0][1]
			perimeter_ch = alpha_list[0][2]

			# compute the percentage change
			delta_area_percent = round(abs(100*((area_cur-area_pre)/area_ch)),2)
			delta_perimeter_percent = round(abs(100*((area_cur-area_pre)/area_ch)),2)

			## check criteria 1 - no significant change
			if delta_area_percent == 0.0 and delta_perimeter_percent == 0.0:

				## criteria 2 - check if there is any inlier points not contained
				gridBool = []
				for in_xy in inlier_part_xy_array:
					gridBool.append(bound_poly.intersects(Point(in_xy[0], in_xy[1])))

				# no outlier points
				if sum(gridBool) == len(inlier_part_xy_array):

					## criteria 3 - check:
					# (1) if any point are close to the boundary -> distance_from_pt_to_boundary <= max_neighbour_distance_of_inlier_points
					gridBool2 = []
					for out_xy in outlier_part_xy_array:
						# within_poly_bool = bound_poly.intersects(Point(out_xy[0], out_xy[1]))

						distance_from_bound = bound_poly.exterior.distance(Point(out_xy[0], out_xy[1]))
						distance_from_bound_bool = distance_from_bound <= max_1NN_dist

						# gridBool2.append(within_poly_bool or distance_from_bound_bool) 
						gridBool2.append(distance_from_bound_bool) 

					# check if there is any points the boundary polygon contains an outlier points 
					# or an outlier point is nearby the polygon boundary, 
					# then change the interior and outlier points and rerun the whole analysis with alpha = 0
					if sum(gridBool2) > 0:

						# change inlier and outlier points
						inlier_part_xy_t = deepcopy(inlier_part_xy)
						outlier_part_xy_t = []
						for idx, grid_bool in enumerate(gridBool2):
							if grid_bool:
								inlier_part_xy_t.append(tuple(outlier_part_xy[idx]))
							else:
								outlier_part_xy_t.append(tuple(outlier_part_xy[idx]))

						del inlier_part_xy, outlier_part_xy
						del inlier_part_xy_array, outlier_part_xy_array
						del max_1NN_dist

						# reset the analysis and start again
						inlier_part_xy = deepcopy(inlier_part_xy_t)
						inlier_part_xy_array = np.array(inlier_part_xy)
						outlier_part_xy = deepcopy(outlier_part_xy_t)
						outlier_part_xy_array = np.array(outlier_part_xy)

						# find maximum neighbour distance among the inlier points
						nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(inlier_part_xy_array)
						distances, indices = nbrs.kneighbors(inlier_part_xy_array)

						# exclude the distance at 1st index, because distance at 1st index == 0 computing average distance from a particle position
						max_1NN_dist = max([loopDist[1] for loopDist in distances])

						# restart
						alpha = 0	# alpha values to consider, start from 0 (i.e. convex hull)
						alpha_list = []
						idx = 0
						continue

					# all fittable points are contained
					elif sum(gridBool2) == 0:

						## criteria 4 and 5 - check internal and external angle
						# compute all min (external or internal) angles created by between three consequtive points
						opt_bound_poly_ext_list = list(bound_poly.exterior.coords)
						pass_criteria_4 = True
						pass_criteria_5 = True
						int_angle_list = []
						for ext_idx in range(1,len(opt_bound_poly_ext_list)-1):

							next_pt = opt_bound_poly_ext_list[ext_idx+1]					
							cur_pt = opt_bound_poly_ext_list[ext_idx]					
							pre_pt = opt_bound_poly_ext_list[ext_idx-1]

							pre_cur = ((cur_pt[0] - pre_pt[0]), (cur_pt[1] - pre_pt[1]))
							next_cur = ((cur_pt[0] - next_pt[0]), (cur_pt[1] - next_pt[1]))

							mag_pre_cur = np.sqrt(pre_cur[0]**2 + pre_cur[1]**2)
							mag_next_cur = np.sqrt(next_cur[0]**2 + next_cur[1]**2)

							dot_product = pre_cur[0]*next_cur[0] + pre_cur[1]*next_cur[1]

							inside = dot_product/(mag_pre_cur*mag_next_cur)

							# int_ext_angle = 0 ~ 180 degrees
							if inside > 1:
								int_ext_angle = 0.0
							elif inside < -1:
								int_ext_angle = 180.0
							else:
								int_ext_angle = round(np.rad2deg(np.arccos(dot_product/(mag_pre_cur*mag_next_cur))),2)

							# encloser boundary -> ~90 then ~90 (tolerance 15)
							if pre_cur[0]*next_cur[1] < pre_cur[1]*next_cur[0]:		# int_ext_angle == external
								int_angle = abs(360-int_ext_angle)
							else:		# int_ext_angle == internal
								int_angle = abs(int_ext_angle)

							ext_angle = 360 - int_angle

							int_angle_list.append(int_angle)

							# minimum angle bend
							if int_angle <= angle_lim or ext_angle <= angle_lim:
								pass_criteria_4 = False

							# # create a 90-90 boundary
							# pre_int_angle = int_angle_list[ext_idx-1]
							# pre_ext_angle = 360 - pre_int_angle
							
							# if abs(pre_ext_angle+ext_angle-180) <= tol_angle:
							# 	pass_criteria_5 = False

						# if pass_criteria_4 and pass_criteria_5:
						# 	opt_alpha_list.append(round(alpha, dp))
							
						# 	alpha_list.append([alpha, area_cur, perimeter_cur])
						# 	alpha += increment
						# 	idx += 1

						# elif pass_criteria_4 == False and pass_criteria_5 == False and len(opt_alpha_list) > 0:
						# 	# just previous one would work
						# 	opt_alpha = round(opt_alpha_list[-1],dp)
							
						# 	alpha_list.append([alpha, area_cur, perimeter_cur])
						# 	alpha += increment
						# 	idx += 1

						# else:
						# 	# store data
						# 	alpha_list.append([alpha, area_cur, perimeter_cur])
						# 	alpha += increment
						# 	idx += 1

						if pass_criteria_4:
							opt_alpha = round(alpha, dp)
							opt_alpha_list.append(round(alpha, dp))
							
							alpha_list.append([alpha, area_cur, perimeter_cur])
							alpha += increment
							idx += 1

						else:
							# store data
							alpha_list.append([alpha, area_cur, perimeter_cur])
							alpha += increment
							idx += 1

				else:
					# store data
					alpha_list.append([alpha, area_cur, perimeter_cur])
					alpha += increment
					idx += 1

			else:
				# store data
				alpha_list.append([alpha, area_cur, perimeter_cur])
				alpha += increment
				idx += 1

	# opt_alpha = alphashape.optimizealpha(inlier_part_xy_array, max_iterations=iter_max, silent=True)
	if opt_alpha == None and len(opt_alpha_list) == 0:
		opt_alpha = 0
	elif opt_alpha == None and len(opt_alpha_list) > 0:
		opt_alpha = round(opt_alpha_list[-1],dp)
		
	opt_bound_poly = alphashape.alphashape(inlier_part_xy_array, opt_alpha)

	if cluster_boundary_buffer > 0:
		opt_bound_poly = opt_bound_poly.buffer(cluster_boundary_buffer, cap_style=1, join_style=3)

	return opt_alpha, opt_bound_poly

def determine_optimal_alpha(points, max_alpha=0.5, min_alpha=0, iter_max=500, dp_accuracy=2):

	### https://alphashape.readthedocs.io/en/latest/_modules/alphashape/optimizealpha.html#optimizealpha
	
	# remove repeating points
	points = list(set(points))

	## check if points would create a line or single point -> optimal alpha value is zero
	points_array = np.array(points)
	if len(points) <= 1 or len(np.unique(points_array[:, 0])) == 1 or len(np.unique(points_array[:,1])) == 1:
		return 0
		
	# find accruacy of up to certain decimal point
	max_alpha_dt_dp = abs(decimal.Decimal(str(max_alpha)).as_tuple().exponent)
	min_alpha_dt_dp = abs(decimal.Decimal(str(min_alpha)).as_tuple().exponent)
	dp_limit = int(max([max_alpha_dt_dp, min_alpha_dt_dp, dp_accuracy]))

	# Set the bounds
	assert min_alpha >= 0 , "The lower bounds must be at least 0"
	# Ensure the upper limit bounds the solution
	assert max_alpha <= sys.float_info.max, (f'The upper bounds must be less than or equal to {sys.float_info.max} on your system')

	lower = min_alpha
	upper = max_alpha

	# user bisection loop to find optimal alpha
	counter = 0
	while (upper - lower) > 0:
		test_alpha = 0.5*(upper + lower)

		# test whether test_alpha is valid
		polygon = alphashape.alphashape(points, test_alpha)
		if isinstance(polygon, Polygon):
			if not isinstance(points, MultiPoint):
				points_multipoints = MultiPoint(list(points))
			alpha_check = all([polygon.intersects(point_mpt) for point_mpt in points_multipoints.geoms])
		elif isinstance(polygon, trimesh.base.Trimesh):
			alpha_check = len(polygon.faces) > 0 and all(trimesh.proximity.signed_distance(polygon, list(points)) >= 0)
		else:
			alpha_check = False

		# update lower and upper bound
		if alpha_check:
			lower = test_alpha
		else:
			upper = test_alpha

		# user default alpha value = 0 if no convergence
		counter += 1
		if counter > iter_max:
			lower = 0
			break
			
		# max alpha value has been reached while only the lower has changed 
		# then the lower bound might have already been good for convergence
		if abs(upper - test_alpha) < 10**(-dp_limit) and upper == max_alpha:
			polygon_min = alphashape.alphashape(points, min_alpha)
			if isinstance(polygon_min, Polygon):
				if not isinstance(points, MultiPoint):
					points_multipoints = MultiPoint(list(points))
				min_alpha_check = all([polygon_min.intersects(point_mpt) for point_mpt in points_multipoints.geoms])
			elif isinstance(polygon_min, trimesh.base.Trimesh):
				min_alpha_check = len(polygon_min.faces) > 0 and all(trimesh.proximity.signed_distance(polygon_min, list(points)) >= 0)
			else:
				min_alpha_check = False
			
			if min_alpha_check:
				lower = min_alpha
				break
			else:
				lower = 0
				break

		# if decimal point accuracy is met
		if round(lower, dp_limit) == round(upper, dp_limit):
			break

	return lower

# determine inlier points to be included in the convexhull to determine the debris cluster boundary
def determine_optimal_CVH_boundary(points, cluster_boundary_buffer=0, output_inlier_points=False): 
	'''
	## determin the optimal alpha value for alpha_shape function
	criteria for determining optimal alpha value (not confirmed but still operational until proven wrong)
	1. The % change of difference in perimeter and area are both nearly equal to zero (0)
			i.e. delta(area%) = 0 and delta(perimeter%) = 0
	2. The boundary should include all the data point. However, if a encompassing data point is considered to be an outlier, 
		then non-inclusion is acceptable. Outlier data point is when the average distance to another data points are 
		1.5 times greater than the average distance between data points that does not include the checking data point
			i.e. if av(d_i)>1.5*av(d_(all other points except the point i)), when point i is checking point
	3. The alpha value does not cause the splitting of data points into two polygons
			i.e. number of polygon generated == 1

	try to find the minimum alpha value that fits the following function
	'''
	
	# remove repeating points
	points = list(set(points))

	## check if points would create a line or single point -> zero cluster area
	points_array = np.array(points)
	if len(points) <= 1 or len(np.unique(points_array[:, 0])) == 1 or len(np.unique(points_array[:,1])) == 1:
		cent_x, cent_y = np.mean(points_array, axis=0)
		return Point(cent_x, cent_y)
		
	####################################################################################################
	### find outlier points
	####################################################################################################

	##################################################
	## find outline locations - based LocalOutliearFactor (LOF) at initial stage
	##################################################
	# LOF
	raw_points_array = np.array(points)
	debris_lof = LocalOutlierFactor(n_neighbors=20)
	outlier_pred = debris_lof.fit_predict(raw_points_array)	# 1 = inlier, -1 = outliers
	outlier_pred_list = outlier_pred.tolist()

	# find all particle positions not considered as an outlier
	inlier_part_xy = [tuple(points[idx]) for idx,pred in enumerate(outlier_pred_list) if pred == 1]
	inlier_part_xy_array = np.array(inlier_part_xy)

	# find maximum neighbour distance among the inlier points
	nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(inlier_part_xy_array)
	distances, indices = nbrs.kneighbors(inlier_part_xy_array)

	# exclude the distance at 1st index, because distance at 1st index == 0 computing average distance from a particle position
	max_1NN_dist = max([loopDist[1] for loopDist in distances])

	# find all particle positions considered as an outlier
	outlier_part_xy = [tuple(points[idx]) for idx,pred in enumerate(outlier_pred_list) if pred == -1]
	outlier_part_xy_array = np.array(outlier_part_xy)
	
	##################################################
	## refine LOF results
	##################################################

	# iteratively change alpha value to find all the inlier points that LOF misplaced
	while True:	

		# perform alpha_shape function with given alpha
		bound_poly = alphashape.alphashape(inlier_part_xy_array, 0)
		if cluster_boundary_buffer > 0:
			bound_poly = bound_poly.buffer(cluster_boundary_buffer, cap_style=1, join_style=3)
			
		# ensure that only one polygon is generated
		try:
			ext_pts_XY = list(bound_poly.exterior.coords)
			if len(ext_pts_XY) == 0:
				bound_poly_error = alphashape.alphashape(raw_points_array, 0)
				if cluster_boundary_buffer > 0:
					bound_poly_error = bound_poly_error.buffer(cluster_boundary_buffer, cap_style=1, join_style=3)

				if output_inlier_points:
					return bound_poly_error, points
				else:
					return bound_poly_error
		except:
			if output_inlier_points:
				return bound_poly, inlier_part_xy
			else:
				return bound_poly

		## criteria - check:
		# (1) if any point are close to the boundary -> distance_from_pt_to_boundary <= max_neighbour_distance_of_inlier_points
		gridBool2 = []
		for out_xy in outlier_part_xy_array:
			distance_from_bound = bound_poly.exterior.distance(Point(out_xy[0], out_xy[1]))
			distance_from_bound_bool = distance_from_bound <= max_1NN_dist
			gridBool2.append(distance_from_bound_bool) 

		# check if there is any points the boundary polygon contains an outlier points 
		# or an outlier point is nearby the polygon boundary, 
		# then change the interior and outlier points and rerun the whole analysis with alpha = 0
		if sum(gridBool2) > 0:

			# change inlier and outlier points
			inlier_part_xy_t = deepcopy(inlier_part_xy)
			outlier_part_xy_t = []
			for idx, grid_bool in enumerate(gridBool2):
				if grid_bool:
					inlier_part_xy_t.append(tuple(outlier_part_xy[idx]))
				else:
					outlier_part_xy_t.append(tuple(outlier_part_xy[idx]))

			del inlier_part_xy, outlier_part_xy
			del inlier_part_xy_array, outlier_part_xy_array
			del max_1NN_dist

			# reset the analysis and start again
			inlier_part_xy = deepcopy(inlier_part_xy_t)
			inlier_part_xy_array = np.array(inlier_part_xy)
			outlier_part_xy = deepcopy(outlier_part_xy_t)
			outlier_part_xy_array = np.array(outlier_part_xy)

			# find maximum neighbour distance among the inlier points
			nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(inlier_part_xy_array)
			distances, indices = nbrs.kneighbors(inlier_part_xy_array)

			# exclude the distance at 1st index, because distance at 1st index == 0 computing average distance from a particle position
			max_1NN_dist = max([loopDist[1] for loopDist in distances])

			# restart
			continue

		# all fittable points are contained
		elif sum(gridBool2) == 0:
			if output_inlier_points:
				return bound_poly, inlier_part_xy
			else:
				return bound_poly

# check whether two clusters merge - concave-hull and convex-hull
def check_merge_v5_0(cluster1, cluster2, merge_overlap_ratio):
	
	if cluster1.boundary_polygon.area == 0 and cluster2.boundary_polygon.area == 0:
		return False
	elif ((cluster1.boundary_polygon.area == 0 and cluster2.boundary_polygon.area > 0) or (cluster1.boundary_polygon.area > 0 and cluster2.boundary_polygon.area == 0)):
		if cluster1.boundary_polygon.intersects(cluster2.boundary_polygon):
			return True
		else:
			return False
			
	# extract boudnary shapely polygon
	poly_cluster1 = cluster1.boundary_polygon
	poly_cluster2 = cluster2.boundary_polygon

	# find if clusters overlap
	overlapping = poly_cluster2.overlaps(poly_cluster1)

	if overlapping == False:
		return False

	else:
		overlapping_region = poly_cluster2.intersection(poly_cluster1)
		overlapping_area = overlapping_region.area

		if ((overlapping_area/poly_cluster1.area) >= merge_overlap_ratio) or ((overlapping_area/poly_cluster2.area) >= merge_overlap_ratio):
			return True
		else:
			return False

###########################################################################
## child functions - energy conservation 
###########################################################################
# no entrainment - E(s) = 1
def SEC_Es0_v5_0(start_xyz, end_xyz, u0, V0, h0, fb, ft, g, dip, part, air=False, fb_0=None, ds=None):

	# DEM data
	x_start = start_xyz[0]
	y_start = start_xyz[1]
	z_start = start_xyz[2]

	x_end = end_xyz[0]
	y_end = end_xyz[1]
	z_end = end_xyz[2]

	# displacement
	DS = np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2 + (z_end - z_start)**2)
	
	# distance travel
	if ds is None:
		# ds = np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2)
		ds = DS

	# erosion ratio (Er) = Es per distance
	# Er = 0 - no entrainment

	# free fall - only KE and GPE - no air friction - no entrainment
	if air:
		try:
			KEf = (u0**2) -  2*g*(z_end - z_start)
			if KEf > 0:
				uf = np.sqrt(KEf)
			else:
				uf = 0.0
		except:
			# convert to zero velocity if error in computation
			uf = 0.0

	# no entrainment - travelling on ground
	else:
		try:
	  
			# vertical acceleration
			if isinstance(dip, (int, float)):
				gz = g*np.cos(dip)
			else:
				gz = g
	
			# pressure energy term
			if part is not None and isinstance(h0, (int, float)):
				ka = (1 - np.sin(np.radians(part.phi)))/(1 + np.sin(np.radians(part.phi)))
				dE_p = 2*(gz*ka*DS*h0*part.div_hi)/(part.rho*V0)
				# dE_p = 0
			elif part is None or h0 is None:
				dE_p = 0

			if isinstance(h0, (int, float)) and ft > 0:
				# compute next velocity - uf_mag - assume u0 = u0_mag
				psi = (g*ft)/(4*h0)

				aa = (1 + 2*psi*ds)
				bb = 4*u0*psi*ds
				cc = 2*g*(z_end - z_start) + 2*ds*( gz*fb + psi*(u0**2) ) - dE_p - (u0**2)
				
				if ((bb**2) - (4*aa*cc)) < 0:
					uf = 0.0
				else: # velocity magnitude should be zero (stationary) or higher
					uf = max(0, (-bb + np.sqrt((bb**2) - (4*aa*cc)))/(2*aa))

			# no depth data - opt-barrier search {assume ft = 0 when h0 = 0}
			elif ft == 0 or h0 is None: 

				if fb_0 is None:
					KEf = (u0**2) + dE_p - 2*g*(z_end - z_start) - 2*gz*ds*fb
				elif isinstance(fb_0, (int, float)) and fb is None:
					KEf = (u0**2) + dE_p - 2*g*(z_end - z_start) - 2*gz*ds*fb_0
				elif fb_0 is None and fb is None:
					KEf = (u0**2) + dE_p - 2*g*(z_end - z_start)
				
				if KEf > 0:
					uf = np.sqrt(KEf)
				else:
					uf = 0.0
		except:
			# convert to zero velocity if error in computation
			uf = 0.0

	return uf, V0


# based on Er ratio value - let E(s) = Es*ds
def SEC_Er_v5_0(start_xyz, end_xyz, u0, V0, h0, fb, ft, Es, g, dip, part, theta_var=5, air=False, fb_0=None, ds=None):

	# DEM data
	x_start = start_xyz[0]
	y_start = start_xyz[1]
	z_start = start_xyz[2]

	x_end = end_xyz[0]
	y_end = end_xyz[1]
	z_end = end_xyz[2]

	# displacement
	DS = np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2 + (z_end - z_start)**2)
	
	# distance travel
	if ds is None:
		# ds = np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2)
		ds = DS

	# travel angle
	np.seterr(invalid='ignore')
	# theta = abs(np.degrees(np.arctan((z_end - z_start) / np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2) )))
	theta = abs(np.degrees(np.arctan((z_end - z_start) / max( 0.001, np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2)) )))

	# free-fall state - automatically no entrainment
	if dip is not None and ((90+theta_var >= theta and theta >= 90-theta_var) or (90+theta_var >= dip and dip >= 90-theta_var) or air):
		return SEC_Es0_v5_0(start_xyz, end_xyz, u0, V0, h0, fb, ft, g, dip, part, air=True, fb_0=fb_0)

	# no entrainment but not necessarily free-fall
	elif Es == 0:
		return SEC_Es0_v5_0(start_xyz, end_xyz, u0, V0, h0, fb, ft, g, dip, part, air=air, fb_0=fb_0)

	# compute next velocity - uf_mag - assume u0 = u0_mag
	try:
		# vertical normal acceleration
		if isinstance(dip, (int, float)):
			gz = g*np.cos(dip)
		else:
			gz = g
   
		# pressure energy term
		if part is not None and isinstance(h0, (int, float)) :
			ka = (1 - np.sin(np.radians(part.phi)))/(1 + np.sin(np.radians(part.phi)))
			dE_p = 2*(gz*ka*DS*h0*part.div_hi)/(part.rho*V0)
			
		elif part is None or h0 is None:
			dE_p = 0		

		if isinstance(h0, (int, float)) and ft > 0:
			# compute next velocity - uf_mag - assume u0 = u0_mag
			psi = (g*ft)/(4*h0)

			aa = Es*ds*(1 + psi*ds)
			bb = 2*u0*psi*Es*(ds**2)
			cc = g*(1+Es*ds)*(z_end - z_start) + Es*(ds**2)*( gz*fb + psi*(u0**2) ) - dE_p - (u0**2)
			
			if ((bb**2) - (4*aa*cc)) < 0:
				uf = 0.0
			else: # velocity magnitude should be zero (stationary) or higher
				uf = max(0, (-bb + np.sqrt((bb**2) - (4*aa*cc)))/(2*aa))
		
		# no depth data - opt-barrier search {assume ft = 0 when h0 = 0}
		elif ft == 0 or h0 is None:
			if fb_0 is None:
				KEf = ((u0**2) + dE_p - g*(1 + Es*ds)*(z_end - z_start) - Es*(ds**2)*fb*gz)/(Es*ds)
			elif isinstance(fb_0, (int, float)) and fb is None:
				KEf = ((u0**2) + dE_p - g*(1 + Es*ds)*(z_end - z_start) - Es*(ds**2)*fb_0*gz)/(Es*ds)
			elif fb_0 is None and fb is None:
				KEf = ((u0**2) + dE_p - g*(1 + Es*ds)*(z_end - z_start) )/(Es*ds)

			if KEf > 0:
				uf = np.sqrt(KEf)
			else:
				uf = 0.0

	except:
		# convert to zero velocity if error in computation
		uf = 0.0

	# compute increased volume
	Vf = V0*Es*ds

	return uf, Vf

# based on Hungr model - let E(s) = exp(Es*ds)
def SEC_Hungr_v5_0(start_xyz, end_xyz, u0, V0, h0, fb, ft, Es, g, dip, part, theta_var=5, air=False, fb_0=None, ds=None):

	# DEM data
	x_start = start_xyz[0]
	y_start = start_xyz[1]
	z_start = start_xyz[2]

	x_end = end_xyz[0]
	y_end = end_xyz[1]
	z_end = end_xyz[2]

	# displacement
	DS = np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2 + (z_end - z_start)**2) 
	
	# distance travel
	if ds is None:
		# ds = np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2)
		ds = DS

	# travel angle
	np.seterr(invalid='ignore')
	# theta = abs(np.degrees(np.arctan((z_end - z_start) / np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2) )))
	theta = abs(np.degrees(np.arctan((z_end - z_start) / max( 0.001, np.sqrt((x_start-x_end)**2 + (y_start-y_end)**2)) )))

	# free-fall state - automatically no entrainment
	if dip is not None and ((90+theta_var >= theta and theta >= 90-theta_var) or (90+theta_var >= dip and dip >= 90-theta_var) or air):
		return SEC_Es0_v5_0(start_xyz, end_xyz, u0, V0, h0, fb, ft, g, dip, part, air=True, fb_0=fb_0)

	# no entrainment but not necessarily free-fall
	elif Es == 0:
		return SEC_Es0_v5_0(start_xyz, end_xyz, u0, V0, h0, fb, ft, g, dip, part, air=air, fb_0=fb_0)

	# compute next velocity - uf_mag - assume u0 = u0_mag
	try:
		# vertical normal acceleration
		if isinstance(dip, (int, float)):
			gz = g*np.cos(dip)
		else:
			gz = g
   
		# pressure energy term
		if part is not None and isinstance(h0, (int, float)) :
			ka = (1 - np.sin(np.radians(part.phi)))/(1 + np.sin(np.radians(part.phi)))
			dE_p = 2*(gz*ka*DS*h0*part.div_hi)/(part.rho*V0)

		elif part is None or h0 is None:
			dE_p = 0	

		if isinstance(h0, (int, float)) and ft > 0:
			# compute next velocity - uf_mag - assume u0 = u0_mag
			psi = (g*ft)/(4*h0)
			E_int = (np.exp(Es*ds) - 1)/Es
			
			aa = np.exp(Es*ds) + 2*psi*E_int
			bb = 4*psi*E_int*u0
			cc = g*(1 + np.exp(Es*ds))*(z_end - z_start) + 2*E_int*( fb*gz + psi*(u0**2) ) - dE_p - (u0**2)
			
			if ((bb**2) - (4*aa*cc)) < 0:
				uf = 0.0
			else: # velocity magnitude should be zero (stationary) or higher
				uf = max(0, (-bb + np.sqrt((bb**2) - (4*aa*cc)))/(2*aa))

		# no depth data - opt-barrier search {assume ft = 0 when h0 = 0}
		elif ft == 0 or h0 is None:
			E_int = (np.exp(Es*ds) - 1)/Es

			if fb_0 is None:
				KEf = ((u0**2) + dE_p - g*(1 + np.exp(Es*ds))*(z_end - z_start) - 2*E_int*gz*fb)/np.exp(Es*ds)
			elif isinstance(fb_0, (int, float)) and fb is None:
				KEf = ((u0**2) + dE_p - g*(1 + np.exp(Es*ds))*(z_end - z_start) - 2*E_int*gz*fb_0)/np.exp(Es*ds)
			elif fb_0 is None and fb is None:
				KEf = ((u0**2) + dE_p - g*(1 + np.exp(Es*ds))*(z_end - z_start))/np.exp(Es*ds)
			
			if KEf > 0:
				uf = np.sqrt(KEf)
			else:
				uf = 0.0

	except:
		# convert to zero velocity if error in computation
		uf = 0.0

	# compute increased volume
	Vf = V0*np.exp(Es*ds)

	return uf, Vf


# compute fb_0 when ft = 0 - for optimal opened-type barrier
def comptue_fb_0_all_cluster_data(f, all_cluster_data):
	# fb_0 = average value of (fb + ft*(u^2)/h) from all the simulations - version 1 - 2022-02-07
	fb_0_list = []
	for cluster_data in all_cluster_data:
		cluster_data_array = np.array(cluster_data)  # 'cID,t,s,x,y,z,u,h,V,D,P,area,CCH_alpha,merged'
		u_array = cluster_data_array[:,6]
		h_array = cluster_data_array[:,7]
		fb_0_i = np.average(f[0] + f[1]*(np.power(u_array,2)/h_array))
		fb_0_list.append(fb_0_i)
	return np.average(fb_0_list)

###########################################################################
## child functions - collision
###########################################################################

# check if point is within any wall boundary
def check_point_in_wall_bound_v1_0(x, y, wall_dict, radius=0, output_full=False):

	if wall_dict == None:
		return False
	
	point = Point(x,y).buffer(radius)

	for wall_id in wall_dict.keys():
		for idx, wall_poly_i in enumerate(wall_dict[wall_id][2]):
			if point.touches(wall_poly_i) or point.intersects(wall_poly_i) or point.within(wall_poly_i):
				if output_full:
					return True, wall_id, idx
				else:
					return True

	# if the loop all ended
	if output_full:
		return False, None, None
	else:
		return False

# check if any particle collided against wall
def check_point_in_wall_bound_mp(wall_col_input):

	part, wall_dict = wall_col_input

	point = Point(part.x,part.y).buffer(part.r)

	for wall_id in wall_dict.keys():
		for idx, wall_poly_i in enumerate(wall_dict[wall_id][2]):
			if point.touches(wall_poly_i) or point.intersects(wall_poly_i) or point.within(wall_poly_i):
				return 1 

	# if the loop all ended
	return 0 

def check_all_part_wall_collision(part_list, wall_dict, max_cpu_num):

	wall_col_input = [(part_i, wall_dict) for part_i in part_list]

	if mp.cpu_count() >= max_cpu_num:
		cpu_num = max_cpu_num 
	else:
		cpu_num = mp.cpu_count()
	pool_part = mp.Pool(cpu_num)

	col_check_list = pool_part.map(check_point_in_wall_bound_mp, wall_col_input)

	pool_part.close()
	pool_part.join()

	if sum(col_check_list) == 0:
		return False   # no collision between wall and particle
	else:
		return True

# collision equation function
def new_part_col_vel(col_mp_input):
	
	part_idx, part_1, part_2_list, col_theta_list, COR_p2p = col_mp_input
	
	## particle - 1
	part_1_m = part_1.Vi*part_1.rho		# mass = volume*density

	### compute contact force - hard sphere COR
	for part_2, col_omega in zip(part_2_list, col_theta_list):		
		
		part_2_m = part_2.Vi*part_2.rho 	# mass = volume*density
		part_2_theta = part_2.travel_direction  # movement direction - theta

		# collision new velocity - part 1 
		v1_fxr = (part_1.ui*np.cos(part_1.travel_direction - col_omega)*(part_1_m - part_2_m*COR_p2p) + (1 + COR_p2p)*part_2_m*part_2.ui*np.cos(part_2_theta - col_omega))/(part_1_m + part_2_m)
		part_1.ux = v1_fxr*np.cos(col_omega) - part_1.ui*np.sin(part_1.travel_direction - col_omega)*np.sin(col_omega)
		part_1.uy = v1_fxr*np.sin(col_omega) + part_1.ui*np.sin(part_1.travel_direction - col_omega)*np.cos(col_omega)
		part_1.ui = np.sqrt(part_1.ux**2 + part_1.uy**2)

		# update new particle travel direction
		part_1.travel_direction = np.arctan2(part_1.uy, part_1.ux)

	return (part_idx, part_1)
	

# collision detection between particles
def part_collision_detection_v5_3(part_list, COR_p2p, cpu_num):
	"""
	detect all collision between particles using discrete collision with one to many collision checking,
	then use colliding distance to find which particles actually collides first
	compute collision only between first contacting particle pairs

	Args:
		part_list (list): list of Particles (class)
		COR_p2p (float): coefficient of restitution from collision (0~1)
		cpu_num (integer): number of CPU threads used in multiprocessing

	Returns:
		successor_part_list (class in list): list of updated Particles (class) after collision
	"""

	# copy particle data
	successor_part_list = deepcopy(part_list)

	#################################
	## one-to-many collision check
	#################################
	# map collision object to particle id number
	geoms = []
	objs = []
	geom_id_to_idx = {}
	for idx,part_i in enumerate(successor_part_list):
		geoms.append(fcl.Sphere(part_i.r))								# FCL sphere geometry
		temp_XYZ = fcl.Transform(np.array([part_i.x, part_i.y, 0.0]))	# FCL sphere XY location
		objs.append(fcl.CollisionObject(geoms[-1], temp_XYZ)) 			# FCL sphere object
		geom_id_to_idx[id(geoms[-1])] = idx 	  						# Create map from geometry IDs to particle idx number 

	# Create manager
	manager = fcl.DynamicAABBTreeCollisionManager()
	manager.registerObjects(objs)
	manager.setup()

	# Create collision request structure
	max_collide_num = len(successor_part_list)*4  # assume max collision is 4*particle number (~ max 4 collision between each particles)
	crequest = fcl.CollisionRequest(num_max_contacts=max_collide_num, enable_contact=True)
	cdata = fcl.CollisionData(crequest, fcl.CollisionResult())
 
	# Run collision request
	manager.collide(cdata, fcl.defaultCollisionCallback)

	# find all colliding particle pairs
	collide_part_F = {}  # part_id: [[other particle id], [collision angle]]
	for dd in cdata.result.contacts:
		part_idx1 = geom_id_to_idx[id(dd.o1)]
		part_idx2 = geom_id_to_idx[id(dd.o2)]

		dir_vector = dd.normal    # contact normal, pointing from object1 to object2
		col_omega = np.arctan2(dir_vector[1], dir_vector[0])%(0.5*np.pi)   # contact angle range: -90 ~ 90
		
		# particle 1
		if part_idx1 not in collide_part_F:
			collide_part_F[part_idx1] = [[part_idx2], [col_omega]]
		else:
			temp_list = deepcopy(collide_part_F[part_idx1])
			temp_list[0].append(part_idx2)
			temp_list[1].append(col_omega)
			collide_part_F[part_idx1] = deepcopy(temp_list)
			del temp_list

		# particle 2
		if part_idx2 not in collide_part_F:
			collide_part_F[part_idx2] = [[part_idx1], [col_omega]]
		else:
			temp_list = deepcopy(collide_part_F[part_idx2])
			temp_list[0].append(part_idx1)
			temp_list[1].append(col_omega)
			collide_part_F[part_idx2] = deepcopy(temp_list)
			del temp_list
   
	# create input library
	col_mp_input = []
	for col_k, col_v  in collide_part_F.items():
		part1 = deepcopy(successor_part_list[col_k])
		part2_list = [deepcopy(successor_part_list[col_k]) for idx in col_v[0]]
		col_mp_input.append((col_k, part1, part2_list, deepcopy(col_v[1]), COR_p2p))

	#################################
	## collision - velocity change
	#################################
	# start multiprocessing
	pool_part = mp.Pool(cpu_num)

	# collision new velocity
	col_data = pool_part.map(new_part_col_vel, col_mp_input)

	# close multiprocessing
	pool_part.close()
	pool_part.join()

	# replace particle
	for col_re in col_data:
		successor_part_list[col_re[0]] = deepcopy(col_re[1])
	   
	return successor_part_list


###########################################################################
## child functions - boundary collision
###########################################################################
# numpy check for boundary collision
def check_boundary_collision_idx(successor_part_list_p1, gridUniqueX, gridUniqueY):
	
	# boundary collision check
	part_x_temp = np.array([part_i.x for part_i in successor_part_list_p1])
	part_y_temp = np.array([part_i.y for part_i in successor_part_list_p1])
	part_idx = np.arange(len(successor_part_list_p1))

	# boundary_collision_idxt = np.where(np.logical_and( np.logical_and(part_x_temp <= gridUniqueX[0], part_x_temp >= gridUniqueX[-1]), np.logical_and(part_y_temp <= gridUniqueY[0], part_y_temp >= gridUniqueY[-1]) ), part_idx, 0)
 
	x_min_idxt = np.where(part_x_temp < gridUniqueX[0], part_idx, np.nan)
	x_max_idxt = np.where(part_x_temp > gridUniqueX[-1], part_idx, np.nan)
	y_min_idxt = np.where(part_y_temp < gridUniqueY[0], part_idx, np.nan)
	y_max_idxt = np.where(part_y_temp > gridUniqueY[-1], part_idx, np.nan)

	total_col_idx = np.unique(np.concatenate((x_min_idxt, x_max_idxt, y_min_idxt, y_max_idxt)))

	boundary_collision_idx = total_col_idx[np.logical_not(np.isnan(total_col_idx))].astype(np.int32)
 
	return boundary_collision_idx

# collision detection between particles
def boundary_CCD_v1_0(boundary_CCD_input):
	"""
	detect all continuous collision between particles with boundary,
	then compute new particle location based on the
	compute collision only between first contacting particle pairs

	Args:
		part_list (list): list of Particles (class)
		COR_p2w (float): coefficient of restitution from collision (0~1) between particle and wall collision
		cpu_num (integer): number of CPU threads used in multiprocessing

	Returns:
		successor_part_list (class in list): list of updated Particles (class) position after collision
	"""

	part, cell_size, t_step, wall_dict, DEM_no_wall, DEM_with_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var, COR_p2w = boundary_CCD_input
	# part, cell_size, t_step, wall_dict, DEM, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var, COR_p2w 

	suc_part = deepcopy(part)
	pre_part = deepcopy(part.predecessor)
 
	#################################
	## check particles outside the boundary
	################################# 
	check_map_bound = [
		(suc_part.x < gridUniqueX[0]),		# collide against min X boundary 
		(suc_part.x > gridUniqueX[-1]),	# collide against max X boundary
		(suc_part.y < gridUniqueY[0]),		# collide against min Y boundary
		(suc_part.y > gridUniqueY[-1])		# collide against max Y boundary
	]

	####################
	## particle data 
	####################
	# previous time step particle position
	x_t0 = pre_part.x
	y_t0 = pre_part.y 
 
	x0 = pre_part.x
	y0 = pre_part.y 
	z0 = pre_part.z

	# current time step particle position
	x_tf = suc_part.x
	y_tf = suc_part.y 

	# assume linear interpolation between particle velocity
	u_t0 = pre_part.ui
	u0 = pre_part.ui
	
	V0 = pre_part.Vi
	h0 = pre_part.hi

	fb = pre_part.fb
	ft = pre_part.ft
	Es = pre_part.Es

	cID = suc_part.clusterID
	mID = suc_part.materialID
 
	dip = pre_part.dip

	# travel angle between current and previous time step
	theta_MD_t0 = suc_part.travel_direction
 
	# particle radius - assume it is equal for all fluid particles
	part_r = suc_part.r

	####################
	## set-up multiple collisions
	####################
	t_CCD_tracking = 0 
	wall_col_count = 0
	final_xy = [0, 0]
	final_travel_angle = 0
	total_travel_dist = 0	
	# count = 0
	while t_CCD_tracking < t_step: 
	
		min_x1_dp = abs(decimal.Decimal(str(x_tf)).as_tuple().exponent)
		min_x2_dp = abs(decimal.Decimal(str(x_t0)).as_tuple().exponent)
		min_y1_dp = abs(decimal.Decimal(str(y_tf)).as_tuple().exponent)
		min_y2_dp = abs(decimal.Decimal(str(y_t0)).as_tuple().exponent)
		round_x_dp = int(0.5*(min_x1_dp+min_x2_dp))
		round_y_dp = int(0.5*(min_y1_dp+min_y2_dp))
		# max_dp = int(max([min_x1_dp, min_x2_dp, min_y1_dp, min_y2_dp]))
		# round_dp = int(0.5*(min_dp + max_dp))

		####################
		## collision type and time 
		####################
		#  determine type of collisions
		bound_col_type = [col_type for col_type,col_bool in enumerate(check_map_bound) if col_bool]	

		# find collision time with boundary
		bound_col_tc = []
		abrupt_break = False
		for col_type_i in bound_col_type:
	
			# if round(x_tf - x_t0, round_x_dp) == 0 and round(y_tf - y_t0, round_y_dp) == 0 and u0 == 0:
			# 	return suc_part
			
			if (round(x_tf - x_t0, round_x_dp) == 0 and col_type_i in [0,1]) or (round(y_tf - y_t0, round_y_dp) == 0 and col_type_i in [2,3]):
				abrupt_break = True
				break

			elif col_type_i == 0:  # collide against min X boundary 
				tc = t_step*(gridUniqueX[0] + part_r - round(x_t0, round_x_dp))/round(x_tf - x_t0, round_x_dp)

			elif col_type_i == 1:  # collide against max X boundary
				tc = t_step*(gridUniqueX[-1] - part_r - round(x_t0, round_x_dp))/round(x_tf - x_t0, round_x_dp)

			elif col_type_i == 2:  # collide against min Y boundary 
				tc = t_step*(gridUniqueY[0] + part_r - round(y_t0, round_y_dp))/round(y_tf - y_t0, round_y_dp)

			elif col_type_i == 3:  # collide against max Y boundary 
				tc = t_step*(gridUniqueY[-1] - part_r - round(y_t0, round_y_dp))/round(y_tf - y_t0, round_y_dp)

			if tc > 1:
				return suc_part
			elif tc < 0:
				tc = 0  

			bound_col_tc.append(tc)
		
		if abrupt_break:
	
			# for certain types of collisions, export new location straight away
			if round(x_tf - x_t0, round_x_dp) == 0 and col_type_i in [0,1]:

				# total time step taken
				t_CCD_tracking = t_step

				# final particle position
				if col_type_i == 0: # x_min
					final_xy[0] = gridUniqueX[0] + abs(x_t0 - gridUniqueX[0])
					final_xy[1] = y_tf
	
				elif col_type_i == 1: # x_max
					final_xy[0] = gridUniqueX[-1] - abs(x_t0 - gridUniqueX[-1]) 
					final_xy[1] = y_tf

				# travelled distance
				total_travel_dist = np.sqrt((final_xy[0] - x_t0)**2 + (y_tf - y_t0)**2) 

				# test
				if theta_MD_t0 >= 0:
					theta_MD_tc = np.pi - abs(theta_MD_t0)
				elif theta_MD_t0 < 0:
					theta_MD_tc = -(np.pi - abs(theta_MD_t0))
				angle_of_incidence = min(abs(theta_MD_tc),  abs(theta_MD_t0))

				# travel angle after collision
				final_travel_angle = theta_MD_tc

				# wall COR based on angle of incidence
				if angle_of_incidence < (np.pi/3):
					wall_col_count += 1

				break
			
			elif round(y_tf - y_t0, round_y_dp) == 0 and col_type_i in [2,3]:
				# total time step taken
				t_CCD_tracking = t_step

				# final particle position
				if col_type_i == 2: # y_min
					final_xy[0] = x_tf
					final_xy[1] = gridUniqueY[0] + abs(y_t0 - gridUniqueY[0])
	
				elif col_type_i == 3: # y_max
					final_xy[0] = x_tf
					final_xy[1] = gridUniqueY[-1] - abs(y_t0 - gridUniqueY[-1]) 

				# travelled distance
				total_travel_dist = np.sqrt((x_tf - x_t0)**2 + (final_xy[1] - y_t0)**2) 

				theta_MD_tc = -theta_MD_t0
	
				if abs(theta_MD_t0) == 0 or abs(theta_MD_t0) == np.pi:
					angle_of_incidence = 0.5*np.pi
				elif abs(theta_MD_t0) >= 0.5*np.pi and abs(theta_MD_t0) < np.pi:
					angle_of_incidence = abs(theta_MD_t0)%(0.5*np.pi)
				elif abs(theta_MD_t0) < 0.5*np.pi and abs(theta_MD_t0) > 0:	
					angle_of_incidence = (0.5*np.pi) - abs(theta_MD_t0)
	
				# travel angle after collision
				final_travel_angle = theta_MD_tc

				# wall COR based on angle of incidence
				if angle_of_incidence < (np.pi/3):
					wall_col_count += 1

				break

		# smallest tc determined
		col_tc = min(bound_col_tc)

		# remaining simulation time
		remaining_time = (t_step - col_tc) - t_CCD_tracking

		# collision location
		x_tc = (col_tc/t_step)*x_tf + (1-(col_tc/t_step))*x_t0
		y_tc = (col_tc/t_step)*y_tf + (1-(col_tc/t_step))*y_t0

		# count number of min_tc for potential simulatenous collision - exactly at the corner
		count_col_tc = bound_col_tc.count(col_tc)

		## reflecting angle
		# corner collision
		if count_col_tc > 1:  

			# reflecting angle for corner - 180 degree turn
			if theta_MD_t0 > 0:
				theta_MD_tc = theta_MD_t0 - np.pi
			elif theta_MD_t0 <= 0:
				theta_MD_tc = theta_MD_t0 + np.pi
	
			angle_of_incidence = 0  

		# boundary collision
		elif count_col_tc == 1:

			# determine the type of collision
			col_type = bound_col_type[bound_col_tc.index(col_tc)]

			# reflecting angle along the y-axes - angle flip
			if col_type in [0,1]:
				if theta_MD_t0 >= 0:
					theta_MD_tc = np.pi - abs(theta_MD_t0)
				elif theta_MD_t0 < 0:
					theta_MD_tc = -(np.pi - abs(theta_MD_t0))

				# angle of incidence - min angle between initial or reflecting angle direction
				angle_of_incidence = min(abs(theta_MD_tc),  abs(theta_MD_t0))

			# reflecting angle along the y-axes - angle flip
			elif col_type in [2,3]:
				theta_MD_tc = -theta_MD_t0

				# angle of incidence - min angle between initial or reflecting angle direction
				if abs(theta_MD_t0) == 0 or abs(theta_MD_t0) == np.pi:
					angle_of_incidence = 0.5*np.pi
				elif abs(theta_MD_t0) >= 0.5*np.pi and abs(theta_MD_t0) < np.pi:
					angle_of_incidence = abs(theta_MD_t0)%(0.5*np.pi)
				elif abs(theta_MD_t0) < 0.5*np.pi and abs(theta_MD_t0) > 0:	
					angle_of_incidence = (0.5*np.pi) - abs(theta_MD_t0)
	
		# reflecting velocity
		if angle_of_incidence < (np.pi/3):
			u_tc = COR_p2w*u_t0
		else:
			u_tc = u_t0	
		# u_tc = COR_p2w*(col_tc*u_tf + (1-col_tc)*u_t0)

		# new particle position after collision
		x_tr = x_tc + remaining_time*u_tc*np.cos(theta_MD_tc)
		y_tr = y_tc + remaining_time*u_tc*np.sin(theta_MD_tc)

		####################
		## check boundary collision for new reflect position
		####################
		check_map_bound = [
			(x_tr < gridUniqueX[0]),	# collide against min X boundary 
			(x_tr > gridUniqueX[-1]),	# collide against max X boundary
			(y_tr < gridUniqueY[0]),	# collide against min Y boundary
			(y_tr > gridUniqueY[-1])	# collide against max Y boundary
		]

		if sum(check_map_bound) == 0 or t_CCD_tracking >= t_step:
			# total time step taken
			t_CCD_tracking = t_step

			# final particle position
			final_xy[0] = x_tr 
			final_xy[1] = y_tr

			# travelled distance
			total_travel_dist += np.sqrt((x_t0 - x_tc)**2 + (y_t0 - y_tc)**2)  # initial collision
			total_travel_dist += np.sqrt((x_tr - x_tc)**2 + (y_tr - y_tc)**2)  # rebound travel

			# travel angle after collision
			final_travel_angle = theta_MD_tc

			# wall COR based on angle of incidence
			if angle_of_incidence < (np.pi/3):
				wall_col_count += 1

			break

		else:
			# count += 1

			# update time step
			t_CCD_tracking += col_tc

			# colliding location
			x_t0 = x_tc
			y_t0 = y_tc

			# next particle location from the colliding 
			x_tf = x_tr
			y_tf = y_tr

			# assume linear interpolation between particle velocity
			u_t0 = u_tc

			# travel angle between current and previous time step
			theta_MD_t0 = theta_MD_tc

			# add to boundary collision
			# wall_col_count += 1 
			if angle_of_incidence < (np.pi/3):
				wall_col_count += 1

			# travelled distance
			total_travel_dist += np.sqrt((x_t0 - x_tc)**2 + (y_t0 - y_tc)**2)  # initial collision
  
	####################
	## find particle datas at final location
	####################
	# local region
	if wall_dict is None:
		local_xy_f, local_z_f = local_cell_v3_0(cell_size, final_xy[0], final_xy[1], DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, z0)
	else:
		local_xy_f, local_z_f = local_cell_v3_0(cell_size, final_xy[0], final_xy[1], DEM_with_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, z0)
	zf = compute_Z_v3_0(final_xy, local_xy_f, local_z_f, interp_method)

	# velocity and volume
	if entrainment_model == 'Hungr':
		uff, Vf = SEC_Hungr_v5_0([x0, y0, z0], [final_xy[0], final_xy[1], zf], u0, V0, h0, fb, ft, Es, g, dip, pre_part, theta_var=Es_theta_var, ds=total_travel_dist)
	elif entrainment_model == 'Er':
		uff, Vf = SEC_Er_v5_0([x0, y0, z0], [final_xy[0], final_xy[1], zf], u0, V0, h0, fb, ft, Es, g, dip, pre_part, theta_var=Es_theta_var, ds=total_travel_dist)

	# check if initial location is within the boundary of a wall
	if wall_dict is not None:
		check_piw = check_point_in_wall_bound_v1_0(final_xy[0], final_xy[1], wall_dict, radius=part_r)
		if check_piw:
			wall_col_count += 1

	# add boundary wall collision effect
	uf = (COR_p2w**(wall_col_count))*uff
	ufx = uf*np.cos(final_travel_angle)
	ufy = uf*np.sin(final_travel_angle)

	####################
	## find successive datas at final location
	####################
	next_part = Particle(final_xy[0], final_xy[1], zf, Vf, uf, None, part_r, cID, None, pre_part)

	# update time
	next_part.update_time(d_time=t_step)

	# distance travelled
	next_part.si = pre_part.si + total_travel_dist

	# material properties
	next_part.materialID  = local_matID_v1_0(final_xy[0], final_xy[1], MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, mID)
	next_part.insert_material(material) 

	next_part.ux = ufx 
	next_part.uy = ufy

	# gradient, dip, dip_direction, travel_direction, dip_travel_direction, travel_curvature
	next_part.compute_dip_and_angles(local_xy_f, local_z_f)

	return next_part

# generate boundary ghost particles for SPH interpolation
# X, Y, radius, Volume
class Ghost_Particle_pre:
	def __init__(self):
		self.Vi = None

class Ghost_Particle:

	def __init__(self, X, Y, r, V, predecessor):

		# XY coordinates
		self.x = X
		self.y = Y
		self.z = None

		# particle radius
		self.r = r

		# volume
		self.Vi = V   
  
		self.predecessor = predecessor

	def update_V(self, V_new):
		self.Vi = V_new
	
	def insert_z(self, z_new):
		self.z = z_new

def boundary_ghost_part_v1_0(gridUniqueX, gridUniqueY, deltaX, deltaY, radius, layers=6):

	# boundary extent
	x_min = gridUniqueX[0] 
	x_max = gridUniqueX[-1] 
	y_min = gridUniqueY[0] 
	y_max = gridUniqueY[-1] 
 
	# new XY grid with ghost
	new_x_grid = np.arange((x_min-0.5*deltaX) - deltaX*layers, (x_max+0.5*deltaX) + deltaX*layers, 2*radius)
	# new_y_grid = np.arange(y_min-deltaY*layers, y_max+deltaY*layers, deltaY)
 
	below_x_min_grid = np.arange((x_min-0.5*deltaX) - deltaX*layers, x_min, 2*radius)
	above_x_max_grid = np.arange(x_max+0.5*deltaX, (x_max+0.5*deltaX) + deltaX*layers, 2*radius)

	below_y_min_grid = np.arange((y_min-0.5*deltaY) - deltaY*layers, y_min, 2*radius)
	above_y_max_grid = np.arange(y_max+0.5*deltaY, (y_max+0.5*deltaY) + deltaY*layers, 2*radius)

	# ghost particle locations
	ghost_part_list = []

	for gx1 in new_x_grid:
		for gy1 in below_y_min_grid:  # below Y_min region
			ghost_part_list.append(Ghost_Particle(gx1, gy1, radius, None, Ghost_Particle_pre()))  # X, Y, radius, Volume, predecessor
		for gy2 in above_y_max_grid:  # above Y_max region
			ghost_part_list.append(Ghost_Particle(gx1, gy2, radius, None, Ghost_Particle_pre()))  # X, Y, radius, Volume, predecessor
   
	for gy3 in gridUniqueY:
		for gx2 in below_x_min_grid:  # below X_min region
			ghost_part_list.append(Ghost_Particle(gx2, gy3, radius, None, Ghost_Particle_pre()))  # X, Y, radius, Volume, predecessor
		for gx3 in above_x_max_grid:  # above X_max region
			ghost_part_list.append(Ghost_Particle(gx3, gy3, radius, None, Ghost_Particle_pre()))  # X, Y, radius, Volume, predecessor

	return ghost_part_list
 
# update ghost particle volume
def update_ghost_particle_volume_MP(mp_input):
	ghost_part, av_V = mp_input
	ghost_part.update_V(av_V)
	return ghost_part

def update_ghost_particle_volume(ghost_part_list, successor_part_list, max_cpu_num):
	
	fluid_V_av = np.mean([part_i.Vi for part_i in successor_part_list])
	update_ghost_V_input = [(g_part, fluid_V_av) for g_part in ghost_part_list]

	# start multiprocessing
	if mp.cpu_count() >= max_cpu_num:
		cpu_num = max_cpu_num 
	else:
		cpu_num = mp.cpu_count()
	pool_part = mp.Pool(cpu_num)

	# collision new velocity
	new_ghost_part_list = pool_part.map(update_ghost_particle_volume_MP, update_ghost_V_input)

	# close multiprocessing
	pool_part.close()
	pool_part.join()

	return new_ghost_part_list


###########################################################################
## child functions - impact pressure of particles
###########################################################################
def part_pressure_Fr_grad_local_MP_v1(part_p_grad_input):

	# sort input
	part_i, g, t_step = part_p_grad_input

	# compute pressure
	part_i.compute_Fr_P(g)
	part_i.compute_grad_local(g, t_step)

	return part_i

###########################################################################
## child functions - particle successor - t_step
###########################################################################

# for wall_dict == None or no particle collision [check_piw_0 == False and check_piw_2 == False]
def no_collision_next_part(part, cell_size_t, t_step, DEM_no_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var):

	# current debris-flow part data
	x0, y0, z0 = part.x, part.y, part.z
	part_radius = part.r
	
	u0 = part.ui
	ux0 = part.ux
	uy0 = part.uy
	
	V0 = part.Vi
	h0 = part.hi

	fb = part.fb
	ft = part.ft
	Es = part.Es

	cID = part.clusterID
	mID = part.materialID
 
	dip = part.dip

	# cell_size and s_step depending on approximity to wall boundary region
	cell_size = cell_size_t[0]

	## iteratively find new position and velocity
	ux_new, uy_new = part.dx_grad_local, part.dy_grad_local
	uf_i = np.sqrt(ux_new**2 + uy_new**2)
	
	# new movement direction
	# uf_dir = np.arctan2(yf-y0, xf-x0)
	uf_dir = np.arctan2(uy_new, ux_new)

	## iteratively find new position and velocity

	iter_num = 0
	iter_max = 5
	uf_error = 0.1

	while True:

		# position - XY
		xf_i = x0 + 0.5*t_step*(ux0 + uf_i*np.cos(uf_dir))
		yf_i = y0 + 0.5*t_step*(uy0 + uf_i*np.sin(uf_dir))

		# elevation - Z
		# local region
		local_xy_f, local_z_f = local_cell_v3_0(cell_size, xf_i, yf_i, DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, z0)
		zf_i = compute_Z_v3_0([xf_i, yf_i], local_xy_f, local_z_f, interp_method)

		# velocity and volume
		if entrainment_model == 'Hungr':
			uf_ii, Vf_ii = SEC_Hungr_v5_0([x0, y0, z0], [xf_i, yf_i, zf_i], u0, V0, h0, fb, ft, Es, g, dip, part, theta_var=Es_theta_var)
		elif entrainment_model == 'Er':
			uf_ii, Vf_ii = SEC_Er_v5_0([x0, y0, z0], [xf_i, yf_i, zf_i], u0, V0, h0, fb, ft, Es, g, dip, part, theta_var=Es_theta_var)

		# first iteration with force-speed slower than energy-speed
		if iter_num == 0 and uf_ii >= uf_i:
			xf, yf, zf = xf_i, yf_i, zf_i
			uf = uf_ii  	# EC-uf
			# uf = uf_i  		# F-uf
			Vf = Vf_ii
			break

		# export converged values  or  exceeded iter_max limit
		elif (abs(uf_ii - uf_i) <= uf_error) or (iter_num >= iter_max):
			xf, yf, zf = xf_i, yf_i, zf_i
			uf = uf_ii  	# EC-uf
			Vf = Vf_ii
			break

		# update step and continue iteration
		else:
			uf_i = uf_ii
			iter_num += 1   	# add iteration number

	# create next particle class
	if uf == 0 and u0 == 0: 
		# X, Y, Z, Vi, ui, hi, radius, clusterID, materialID, predecessor
		next_part = Particle(x0, y0, z0, V0, 0.0, None, part_radius, cID, mID, part)
	else:
		next_part = Particle(xf, yf, zf, Vf, uf, None, part_radius, cID, None, part)

	# update time
	next_part.update_time(d_time=t_step)

	# distance travelled
	next_part.compute_s()

	# material properties
	materialID = local_matID_v1_0(xf, yf, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, mID)
	next_part.materialID = materialID
	next_part.insert_material(material) 

	next_part.ux = uf*np.cos(uf_dir)  
	next_part.uy = uf*np.sin(uf_dir)

	# gradient, dip, dip_direction, travel_direction, dip_travel_direction, travel_curvature
	next_part.compute_dip_and_angles(local_xy_f, local_z_f)

	return next_part


def wall_bound_line_intersection_check(wall_dict, x0, y0, xf, yf, output_full=True):
	
	point_travel_path = LineString([(x0, y0), (xf, yf)])

	# iterate through each wall segment and check for particle travel path and wall collision
	for wall_id in wall_dict.keys():
		for idx, wall_poly_i in enumerate(wall_dict[wall_id][2]):
			poly_line = LineString(list(wall_poly_i.exterior.coords))
			if point_travel_path.intersects(poly_line):
				if output_full:
					return True, wall_id, idx
				else:
					return True

	# if the loop all ended
	if output_full:
		return False, None, None
	else:
		return False
 

def wall_bound_line_intersection(wall_dict, wall_id, seg_id, x0, y0, xf, yf, bd_top=False):

	if wall_id is None and seg_id is None:
		return (xf, yf)

	# wall polygon and travel path of particle
	wall_poly_i = wall_dict[wall_id][2][seg_id]
	point_travel_path = LineString([(x0, y0), (xf, yf)])

	# intersection point
	poly_line = LineString(list(wall_poly_i.exterior.coords))
	intersection_pt = point_travel_path.intersection(poly_line)

	if point_travel_path.intersects(poly_line) == False:

		# cross_x, cross_y = xf, yf
  
		polygon_0_dist = wall_poly_i.exterior.distance(Point(x0, y0))
		polygon_f_dist = wall_poly_i.exterior.distance(Point(xf, yf))
  
		# select point closest to the wall polygon from initial
		if bd_top == False:
			if polygon_0_dist <= polygon_f_dist:
				cross_x, cross_y = x0, y0
			elif polygon_0_dist > polygon_f_dist:
				cross_x, cross_y = xf, yf
			
		# select point closest to the wall polygon from final
		elif bd_top == True:
			if polygon_0_dist > polygon_f_dist:
				cross_x, cross_y = x0, y0
			elif polygon_0_dist <= polygon_f_dist:
				cross_x, cross_y = xf, yf

	# if only one interaction, then output such
	elif intersection_pt.geom_type == "Point":
		cross_x, cross_y = float(intersection_pt.x), float(intersection_pt.y)
	
	# if multiple intersection, choose the closer one
	elif intersection_pt.geom_type == "MultiPoint":

		# distance from (x0, y0)
		each_point_list = list(intersection_pt.geoms)
		dist_from_0_intersection_pt = []
		intersection_pt_xy = []
		for pp in each_point_list:
			dist_from_0_intersection_pt.append(np.sqrt((x0 - pp.x)**2 + (y0 - pp.y)**2))
			intersection_pt_xy.append((float(pp.x), float(pp.y)))

		# min dist from initial
		if bd_top == False:
			min_dist_idx = dist_from_0_intersection_pt.index(min(dist_from_0_intersection_pt))
			cross_x, cross_y = intersection_pt_xy[min_dist_idx]

		# max dist from initial
		elif bd_top == True:
			max_dist_idx = dist_from_0_intersection_pt.index(max(dist_from_0_intersection_pt))
			cross_x, cross_y = intersection_pt_xy[max_dist_idx]

	# a line of interaction with the wall
	elif intersection_pt.geom_type == "LineString":
		
		# distance from (x0, y0)
		each_point_list = list(intersection_pt.coords)
		dist_from_0_intersection_pt = []
		intersection_pt_xy = []
		for pp in each_point_list:
			dist_from_0_intersection_pt.append(np.sqrt((x0 - pp[0])**2 + (y0 - pp[1])**2))
			intersection_pt_xy.append((float(pp[0]), float(pp[1])))

		# min dist from initial
		if bd_top == False:
			min_dist_idx = dist_from_0_intersection_pt.index(min(dist_from_0_intersection_pt))
			cross_x, cross_y = intersection_pt_xy[min_dist_idx]
		# max dist from initial
		elif bd_top == True:
			max_dist_idx = dist_from_0_intersection_pt.index(max(dist_from_0_intersection_pt))
			cross_x, cross_y = intersection_pt_xy[max_dist_idx]

	return (cross_x, cross_y)


def generate_part_successors_mp_v18_0(mp_input_suc): 

	# sort input data
	part, cell_size_t, t_step, wall_dict, DEM_no_wall, DEM_with_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, g, material, COR_p2w, entrainment_model, Es_theta_var = mp_input_suc

	# wall_info = [slit_ratio, wall_segment_number, P_or_V ('P' or 'V'), wall_segment_oriP (-90 ~ 90), wall_oriP (-90 ~ 90), thickness, length, Z_opt (1~4), h_or_z, central_X_coord, central_Y_coord]
	# key = wall_id, value = [(overall) wall_info, [each wall section data], [each wall section shapely polygon], overall_wall_multipolygon]

	###########################################
	## no possible wall-particle collision
	###########################################
	if wall_dict == None:
		# generate next particle
		return no_collision_next_part(part, cell_size_t, t_step, DEM_no_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var)

	###########################################
	## potential wall-particle collision
	###########################################
	else:

		try: 

			### current debris-flow part data
			x0, y0, z0 = part.x, part.y, part.z
			part_radius = part.r

			mID0 = part.materialID
			
			u0 = part.ui
			ux0 = part.ux
			uy0 = part.uy
			
			V0 = part.Vi
			h0 = part.hi

			fb = part.fb
			ft = part.ft
			Es = part.Es

			cID = part.clusterID
			mID = part.materialID
	  
			dip = part.dip

			###########################################
			## initial position collidion check
			###########################################
			# check if initial location is within the boundary of a wall
			check_piw_0, wall_id_0, seg_id_0 = check_point_in_wall_bound_v1_0(x0, y0, wall_dict, radius=part_radius, output_full=True)

			###########################################
			## determine local cell size and potential of collision at final state
			###########################################
			# current particle poisition as shapely point
			part_Point = Point(x0, y0).buffer(part_radius)

			# cell_size and s_step depending on approximity to wall boundary region
			wall_bound = None
			for wall_id in wall_dict.keys():

				wall_multi_poly = wall_dict[wall_id][-1]
				cur_wall_bound = wall_multi_poly.bounds

				if wall_bound == None:
					wall_bound = list(cur_wall_bound)
				else:
					# x_min and y_min
					wall_bound[0] = min(wall_bound[0], cur_wall_bound[0])
					wall_bound[1] = min(wall_bound[1], cur_wall_bound[1])

					# x_max and y_max
					wall_bound[2] = max(wall_bound[2], cur_wall_bound[2])
					wall_bound[3] = max(wall_bound[3], cur_wall_bound[3])

			# wall boundary region + buffer shapely polygon
			wall_bound_poly = Polygon([
				(wall_bound[0], wall_bound[1]),  		# x_min, y_min
				(wall_bound[2], wall_bound[1]),  		# x_max, y_min
				(wall_bound[2], wall_bound[3]),  		# x_max, y_max
				(wall_bound[0], wall_bound[3])   		# x_min, y_max
			]).buffer(max(cell_size_t), join_style=2)		# buffer, mitre joined

			# check whether the particle is likely to interact with any wall
			if part_Point.touches(wall_bound_poly) or part_Point.intersects(wall_bound_poly):
				cell_size = cell_size_t[1]
				# check_piw_1 = True
			else:
				cell_size = cell_size_t[0]
				# check_piw_1 = False

			###########################################
			## no initial colliding of particles
			###########################################
			if check_piw_0 == False:

				# find new position and velocity based on force
				ux_new, uy_new = part.dx_grad_local, part.dy_grad_local
				uf = np.sqrt(ux_new**2 + uy_new**2)
				uf_dir = np.arctan2(uy_new, ux_new)

				# position - XY
				xf = x0 + 0.5*t_step*(ux0 + ux_new)
				yf = y0 + 0.5*t_step*(uy0 + uy_new)

				# check if initial location is within the boundary of a wall
				check_piw_2, wall_id_2, seg_id_2 = wall_bound_line_intersection_check(wall_dict, x0, y0, xf, yf, output_full=True)
				check_piw_22 = check_point_in_wall_bound_v1_0(xf, yf, wall_dict, radius=part_radius, output_full=False)

				# no collision
				if check_piw_2 == False and check_piw_22 == False:
					# generate next particle
					# return no_collision_next_part(part, cell_size_t, t_step, DEM_no_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var)
					return no_collision_next_part(part, cell_size_t, t_step, DEM_with_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var)

				# wall collision
				# horizontal movement then vertical movement
				elif check_piw_2 == True or check_piw_22 == True:

					# DEM elevation without wall
					local_xy_dem, local_z_dem = local_cell_v3_0(cell_size, xf, yf, DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
					dem_z = compute_Z_v3_0([xf, yf], local_xy_dem, local_z_dem, interp_method)

					# wall elevation
					if check_piw_22 == True and check_piw_2 == False:
						check_piw_22, wall_id_2, seg_id_2 = check_point_in_wall_bound_v1_0(xf, yf, wall_dict, radius=part_radius, output_full=True)
	 
					if wall_dict[wall_id_2][1][seg_id_2][2] == 1:
						z_top = wall_dict[wall_id_2][1][seg_id_2][3]
					else:
						z_top = dem_z + wall_dict[wall_id_2][1][seg_id_2][3]

					## colliding point
					# determine contact point of wall boundary and particle movement
					cross_x, cross_y = wall_bound_line_intersection(wall_dict, wall_id_2, seg_id_2, x0, y0, xf, yf)

					# time of collision
					t_norm_col_x = (cross_x - x0)/(xf - x0)
					# t_norm_b_col_x = (cross_x - x0 - 0.5*deltaX)/(xf - x0)

					t_norm_col_y = (cross_y - y0)/(yf - y0)
					# t_norm_b_col_y = (cross_y - x0 - 0.5*deltaY)/(yf - y0)

					t_norm_col = 0.5*(t_norm_col_x + t_norm_col_y)
					# t_norm_b_col = min((t_norm_b_col_x, t_norm_b_col_y))
					t_norm_b_col = t_norm_col - t_step*0.1

					# location just before wall collision
					just_before_x = x0 + (xf - x0)*t_norm_b_col
					just_before_y = y0 + (yf - y0)*t_norm_b_col

					# at colliding elevation (z)
					local_xy_c, local_z_c = local_cell_v3_0(cell_size, cross_x, cross_y, DEM_with_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, z0)
					z_cross = compute_Z_v3_0([cross_x, cross_y], local_xy_c, local_z_c, interp_method)
	 
					# local_xy_b_c, local_z_b_c = local_cell_v3_0(cell_size, just_before_x, just_before_y, DEM_with_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, z0)
					local_xy_b_c, local_z_b_c = local_cell_v3_0(cell_size, just_before_x, just_before_y, DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, z0)
					z_b_cross = compute_Z_v3_0([just_before_x, just_before_y], local_xy_b_c, local_z_b_c, interp_method)

					# velocity and volume - just before collision
					if entrainment_model == 'Hungr':
						u_b_cross, V_b_cross = SEC_Hungr_v5_0([x0, y0, z0], [just_before_x, just_before_y, z_b_cross], u0, V0, h0, fb, ft, Es, g, dip, part, theta_var=Es_theta_var)
					elif entrainment_model == 'Er':
						u_b_cross, V_b_cross = SEC_Er_v5_0([x0, y0, z0], [just_before_x, just_before_y, z_b_cross], u0, V0, h0, fb, ft, Es, g, dip, part, theta_var=Es_theta_var)

					# max height reachable from energy conservation model
					z_max_reachable = z_b_cross + ((COR_p2w*u0)**2)/(2*g)

					# elevation distance travelled
					# zf = z_b_cross + u_b_cross*COR_p2w*(1-t_norm_col)*t_step 
					zf = z_b_cross + u_b_cross*COR_p2w*(1-t_norm_col)*t_step - 0.5*g*((1-t_norm_col)*t_step)**2

					if zf >= min(z_max_reachable, z_top):
						zf = min(z_max_reachable, z_top)

					# velocity and volume - just before during wall travel
					# input:  start_xyz, end_xyz, u0, V0, h0, fb, ft, g, dip, part, air=False, fb_0=None, ds=None
					uf, Vf = SEC_Es0_v5_0([just_before_x, just_before_y, z_b_cross], [cross_x, cross_y, zf], u_b_cross*COR_p2w, V_b_cross, None, None, None, g, None, None, air=True, fb_0=None, ds=None)

					# material properties ID
					materialID = local_matID_v1_0(cross_x, cross_y, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, mID0)

					# create next particle class
					next_part = Particle(cross_x, cross_y, zf, Vf, uf, None, part_radius, cID, materialID, part)

					# select wall_climbing or on top of wall
					next_part.wall = True
	 
					# update time
					next_part.update_time(d_time=t_step)

					# distance travelled
					next_part.compute_s()

					# material properties
					next_part.insert_material(material) 

					next_part.ux = uf*np.cos(uf_dir)  
					next_part.uy = uf*np.sin(uf_dir)

					# gradient, dip, dip_direction, travel_direction, dip_travel_direction, travel_curvature
					next_part.compute_dip_and_angles(local_xy_c, local_z_c)

					return next_part

			###########################################
			## initial colliding of particles
			###########################################
			# no collision at start, but within wall region, check for collision at final state
			elif check_piw_0 == True:
				
				# check whether the particle is inside or just outside the wall
				check_piw_0_pnw_dist = wall_dict[wall_id_0][2][seg_id_0].exterior.distance(Point(x0, y0))
				# check_piw_00 = check_point_in_wall_bound_v1_0(x0, y0, wall_dict, radius=0, output_full=False) 

				# DEM elevation without wall
				local_xy_dem, local_z_dem = local_cell_v3_0(cell_size, x0, y0, DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
				dem_z = compute_Z_v3_0([x0, y0], local_xy_dem, local_z_dem, interp_method)
				
				# wall elevation
				if wall_dict[wall_id_0][1][seg_id_0][2] == 1:
					z_top = wall_dict[wall_id_0][1][seg_id_0][3]
				else:
					z_top = dem_z + wall_dict[wall_id_0][1][seg_id_0][3]

				# travel direction continues from the predecessor - before the wall collision
				uf_dir = part.predecessor.travel_direction
	
				# max height reachable from energy conservation model
				z_max_reachable = z0 + (u0**2)/(2*g)
				s_max_reachable = (u0**2)/g

				# particle elevation less than wall_top_elevation -> on the wall side
				# particle very close to the wall exterior boundary -> distance less than 0.25*cell size
				if check_piw_0_pnw_dist > part_radius:
					return no_collision_next_part(part, cell_size_t, t_step, DEM_no_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var)

				elif z0 < z_top and check_piw_0_pnw_dist <= part_radius:

					# elevation change from velocity
					# if u0 == 0: # stationary is the air
					if abs(u0) <= 0.1: # stationary is the air
						# zf = z0 - 0.5*t_step*np.sqrt(2*g*abs(z0 - part.predecessor.z))
						zf = z0 - 0.5*g*(t_step**2) 	# based on SUVAT
					elif (z0 - part.predecessor.z) <= 0:  # going downwards
						# zf = z0 - u0*t_step
						zf = z0 - u0*t_step - 0.5*g*(t_step**2)   # based on SUVAT
					elif u0 > 0 and (z0 - part.predecessor.z) > 0:  # positive speed to still go up
						# zf = z0 + u0*t_step
						zf = z0 + u0*t_step - 0.5*g*(t_step**2)   # based on SUVAT

					# limit of elevaiton travel
					if zf >= min(z_max_reachable, z_top):
						zf = min(z_max_reachable, z_top)
					# if zf >= z_max_reachable:
					# 	zf = z_max_reachable
					elif zf <= dem_z:
						zf = dem_z

					# speed from elevation change
					uf, Vf = SEC_Es0_v5_0([x0, y0, z0], [x0, y0, zf], u0, V0, None, None, None, g, None, None, air=True, fb_0=None, ds=None)

					# speed loss from collision with ground
					if zf <= dem_z:
						uf = COR_p2w*uf

					# create next particle class
					next_part = Particle(x0, y0, zf, Vf, uf, None, part_radius, cID, mID, part)
	 
					# select wall_climbing or on top of wall
					next_part.wall = True

					# update time
					next_part.update_time(d_time=t_step)

					# distance travelled
					next_part.compute_s()

					# material properties
					next_part.insert_material(material) 

					next_part.ux = uf*np.cos(uf_dir)  
					next_part.uy = uf*np.sin(uf_dir)  

					# gradient, dip, dip_direction, travel_direction, dip_travel_direction, travel_curvature
					local_xy_c, local_z_c = local_cell_v3_0(cell_size, x0, y0, DEM_with_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, z0)
					next_part.compute_dip_and_angles(local_xy_c, local_z_c)

					return next_part

				# particle elevation more than or equal to wall_top_elevation -> on wall top
				elif z0 >= z_top or check_piw_0_pnw_dist > part_radius: # (dem_z > z_top and z0 < z_top and check_piw_0_pnw_dist > part_radius):
		
					#### assume parabolic movement
					# position - XY
					xf = x0 + t_step*u0*np.cos(uf_dir)  
					yf = y0 + t_step*u0*np.sin(uf_dir)
	 
					local_xyf_dem, local_zf_dem = local_cell_v3_0(cell_size, xf, yf, DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
					dem_zf = compute_Z_v3_0([xf, yf], local_xyf_dem, local_zf_dem, interp_method)

					# elevation change
					# if u0 == 0: # stationary is the air
					if abs(u0) <= 0.1: # almost stationary is the air
						# zf = z0 - 0.5*t_step*np.sqrt(2*g*abs(z0 - part.predecessor.z))
						zf = z0 - 0.5*g*(t_step**2) 	# based on SUVAT
					elif (z0 - part.predecessor.z) <= 0:  # going downwards
						# zf = z0 - u0*t_step
						zf = z0 - u0*t_step - 0.5*g*(t_step**2)   # based on SUVAT
					elif u0 > 0 and (z0 - part.predecessor.z) > 0:  # positive speed to still go up
						# zf = z0 + u0*t_step
						zf = z0 + u0*t_step - 0.5*g*(t_step**2)   # based on SUVAT
	  
					# limit of minimum elevaiton travel and maximum travel distance
					if zf <= dem_zf or np.sqrt((xf-x0)**2 + (yf-y0)**2) >= s_max_reachable:
						zf = dem_zf
						COR_p2w_t = COR_p2w
					# limit of elevaiton travel and
					elif zf >= z_max_reachable and np.sqrt((xf-x0)**2 + (yf-y0)**2) < s_max_reachable:
						zf = z_max_reachable
						COR_p2w_t = 0.01
					# speed loss from collision with ground
					elif zf > dem_zf and zf < z_max_reachable and np.sqrt((xf-x0)**2 + (yf-y0)**2) < s_max_reachable:
						COR_p2w_t = 1
	  
					# speed from elevation change
					uf, Vf = SEC_Es0_v5_0([x0, y0, z0], [xf, yf, zf], u0, V0, None, None, None, g, None, None, air=True, fb_0=None, ds=None)

					# create next particle class
					next_part = Particle(xf, yf, zf, Vf, COR_p2w_t*uf, None, part_radius, cID, mID, part)
	 
					# select wall_climbing or on top of wall
					next_part.wall = True

					# update time
					next_part.update_time(d_time=t_step)

					# distance travelled
					next_part.compute_s()

					# material properties
					next_part.insert_material(material) 

					next_part.ux = COR_p2w_t*uf*np.cos(uf_dir)  
					next_part.uy = COR_p2w_t*uf*np.sin(uf_dir)  

					# gradient, dip, dip_direction, travel_direction, dip_travel_direction, travel_curvature
					local_xy_c, local_z_c = local_cell_v3_0(cell_size, xf, yf, DEM_with_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, z0)
					next_part.compute_dip_and_angles(local_xy_c, local_z_c)
						
					return next_part

		except Exception as e:
		
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

			print("error cuased by: ")
			print(exc_type, fname, exc_tb.tb_lineno)
			print("terminated early")
			print("error cuased by: "+str(e))
			print()

			return None
			
###########################################################################
## child functions - barrier performance
###########################################################################
# function to assess whether part is inside the polygon
def part_in_region(mp_input_PinR):
	part_id, part_class, region_poly = mp_input_PinR
	if Point(part_class.x, part_class.y).within(region_poly):
		return part_id		
	else:
		return None

# wall_region
def compute_wall_surrounding_region_v2_0(wall_dict, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, cell_size, interp_method, SPH=False):
	'''
	generate wall_dict containing all objects to be categorized as buildings or barriers
	region boundary before and after the barrier/building
	'''

	#############################################
	## region before and after the wall
	#############################################
	wall_bound_region = {}		# key = wall_id, value = [shapely polygon before, shapely polygon after]
	for wall_id in wall_dict.keys():

		# wall overall polygon data for particular wall_id
		wall_multi_poly = wall_dict[wall_id][-1] 

		###################
		## step 0 - get overall wall group information
		###################
		## generate bound box of the barrier/building group
		# find all the shapely polygon exterior points in the multi-polygon
		# then create a shapely multi-point
		all_wall_xt_pt = []
		for wall_poly in wall_multi_poly.geoms:
			wall_poly_ext = list(wall_poly.exterior.coords)
			for ext_xy in wall_poly_ext:
				all_wall_xt_pt.append((ext_xy[0], ext_xy[1]))
		wall_multi_ext_pt = MultiPoint(all_wall_xt_pt)

		# assuming that the wall creates a rectangular region, 
		# find the minimum box region that contains all the shapely polygon exterior points
		cur_wall_bound_box = wall_multi_ext_pt.minimum_rotated_rectangle

		## compute length and width of the rectangle
		# get coordinates of polygon vertices of boundary box
		corner_x, corner_y = cur_wall_bound_box.exterior.coords.xy

		# get distance between bounding box
		edge_dist = (Point(corner_x[0], corner_y[0]).distance(Point(corner_x[1], corner_y[1])), Point(corner_x[1], corner_y[1]).distance(Point(corner_x[2], corner_y[2])))

		# length and width
		bound_length = max(edge_dist)
		bound_width = min(edge_dist)

		## wall overall centroid
		wall_centroid = cur_wall_bound_box.centroid
		wall_cx = wall_centroid.x
		wall_cy = wall_centroid.y

		## wall overall orientation
		# follow the direction of the longer axis (i.e. length)
		# NOTE: -90 < wall_oriP_deg <= 90
		# wall_oriP_deg 

		# find dY and dX from one bound point to other - arctan2(dy, dx) - along the region length
		if bound_length == bound_width or bound_length == edge_dist[0]:   
			# if region is square, then it doesn't matter which orientation it is facing
			wall_oriP_deg_t = np.rad2deg(np.arctan2((corner_y[1] - corner_y[0]), (corner_x[1] - corner_x[0])))

		elif bound_length == edge_dist[1]:
			wall_oriP_deg_t = np.rad2deg(np.arctan2((corner_y[2] - corner_y[1]), (corner_x[2] - corner_x[1])))

		# wall_oriP_deg_t ranges from -180 to 180
		# fit the wall_oriP_deg_t so that: -90 < wall_oriP_deg <= 90
		if wall_oriP_deg_t <= 90 and wall_oriP_deg_t > -90:  # fits the region
			wall_oriP_deg = wall_oriP_deg_t

		elif wall_oriP_deg_t <= 180 and wall_oriP_deg_t > 90: 	# 90 < wall_oriP_deg_t <= 180
			wall_oriP_deg = wall_oriP_deg_t - 180

		elif wall_oriP_deg_t <= -90 and wall_oriP_deg_t >= -180:   # -180 <= wall_oriP_deg_t <= -90
			wall_oriP_deg = wall_oriP_deg_t + 180

		## wall before and after region orientation
		if wall_oriP_deg >= 0:
			region_oriP_deg = 90 + wall_oriP_deg
		elif wall_oriP_deg < 0:
			region_oriP_deg = wall_oriP_deg - 90

		###################
		## step 1 - find center coordinate 
		###################
		if SPH:  # include wall region down the middle and check larger (2*bound_length x 2*bound_length) region - for SPH depth computation
			# before wall centroid
			reg1_cx = wall_cx - (0.5*bound_length)*np.cos(np.deg2rad(region_oriP_deg))
			reg1_cy = wall_cy - (0.5*bound_length)*np.sin(np.deg2rad(region_oriP_deg))

			# after wall centroid
			reg2_cx = wall_cx + (0.5*bound_length)*np.cos(np.deg2rad(region_oriP_deg))
			reg2_cy = wall_cy + (0.5*bound_length)*np.sin(np.deg2rad(region_oriP_deg))

			# generate box for before and after region
			reg1_temp_poly = box(reg1_cx-0.5*bound_length, reg1_cy-0.5*bound_length, reg1_cx+0.5*bound_length, reg1_cy+0.5*bound_length)
			reg2_temp_poly = box(reg2_cx-0.5*bound_length, reg2_cy-0.5*bound_length, reg2_cx+0.5*bound_length, reg2_cy+0.5*bound_length)

		else:	# exclude wall region and only check (bound_length x bound_length) region - for wall performance
			# before wall centroid
			reg1_cx = wall_cx - (0.5*bound_length + 0.5*bound_width)*np.cos(np.deg2rad(region_oriP_deg))
			reg1_cy = wall_cy - (0.5*bound_length + 0.5*bound_width)*np.sin(np.deg2rad(region_oriP_deg))

			# after wall centroid
			reg2_cx = wall_cx + (0.5*bound_length + 0.5*bound_width)*np.cos(np.deg2rad(region_oriP_deg))
			reg2_cy = wall_cy + (0.5*bound_length + 0.5*bound_width)*np.sin(np.deg2rad(region_oriP_deg))

			# generate box for before and after region
			reg1_temp_poly = box(reg1_cx-0.5*bound_length, reg1_cy-0.5*bound_length, reg1_cx+0.5*bound_length, reg1_cy+0.5*bound_length)
			reg2_temp_poly = box(reg2_cx-0.5*bound_length, reg2_cy-0.5*bound_length, reg2_cx+0.5*bound_length, reg2_cy+0.5*bound_length)

		# rotate box to match the wall orientation
		reg1_poly = rotate(reg1_temp_poly, wall_oriP_deg)
		reg2_poly = rotate(reg2_temp_poly, wall_oriP_deg)

		## determine the before and after region based on the elevation of the region -> before elevation > after elevation
		## region 1 - corners and central point
		reg1_bound_z_av = 0

		# corner points
		reg1_bound_x, reg1_bound_y = reg1_poly.exterior.coords.xy
		for reg1_x, reg1_y in zip(reg1_bound_x, reg1_bound_y):
			local_xy, local_z = local_cell_v3_0(cell_size, reg1_x, reg1_y, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
			reg1_bound_z_av += compute_Z_v3_0([reg1_x, reg1_y], local_xy, local_z, interp_method)  
			del local_xy, local_z

		# center point
		local_xy, local_z = local_cell_v3_0(cell_size, reg1_cx, reg1_cy, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
		reg1_bound_z_av += compute_Z_v3_0([reg1_cx, reg1_cy], local_xy, local_z, interp_method)  
		del local_xy, local_z

		# average region elevation
		reg1_bound_z_av = reg1_bound_z_av/5

		## region 2
		reg2_bound_z_av = 0

		# corner points
		reg2_bound_x, reg2_bound_y = reg2_poly.exterior.coords.xy
		for reg2_x, reg2_y in zip(reg2_bound_x, reg2_bound_y):
			local_xy, local_z = local_cell_v3_0(cell_size, reg2_x, reg2_y, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
			reg2_bound_z_av += compute_Z_v3_0([reg2_x, reg2_y], local_xy, local_z, interp_method)  
			del local_xy, local_z

		# center point
		local_xy, local_z = local_cell_v3_0(cell_size, reg2_cx, reg2_cy, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
		reg2_bound_z_av += compute_Z_v3_0([reg2_cx, reg2_cy], local_xy, local_z, interp_method)  
		del local_xy, local_z

		# average region elevation
		reg2_bound_z_av = reg2_bound_z_av/5

		# store before and after region
		if reg1_bound_z_av >= reg2_bound_z_av:
			wall_bound_region[wall_id] = [reg1_poly, reg2_poly]		# [before, after] wall
		elif reg1_bound_z_av < reg2_bound_z_av:
			wall_bound_region[wall_id] = [reg2_poly, reg1_poly]		# [before, after] wall

	# wall_bound_region  key = wall_id, value = [shapely polygon before, shapely polygon after]
	return wall_bound_region

# compute speed reduction (SR) and trap ratio (TR)
def compute_performance_v2_0(wall_perform_region, all_part_list, max_cpu_num):

	# add wall data into topography and as dictionary
	# wall_info = [slit_ratio, wall_segment_number, P_or_V ('P' or 'V'), wall_segment_oriP (-90 ~ 90), wall_oriP (-90 ~ 90), wall_segment_thickness, wall_length, Z_opt (1~4), h_or_z, central_X_coord, central_Y_coord]
	# wall_dict -> key = wall_id, value = [(overall) wall_info, [each wall section data], [each wall section shapely polygon], sheply all wall multipolygon]
	
	# wall_perform_region = {}		# key = wall_id, value = [shapely polygon before, shapely polygon after]

	#############################################
	## compute particle velocity and volume before and after the wall
	#############################################
	# multiprocessing set-up
	if mp.cpu_count() >= max_cpu_num:
		cpu_num = max_cpu_num 
	else:
		cpu_num = mp.cpu_count()
	pool_PinR = mp.Pool(cpu_num)

	# check which particles are contained in the region
	wall_perform_time = {}	# key = wall_id, value = [before[part_id, part]per time_step, after[part_id, part]per time_step] 
	for wall_id in wall_perform_region.keys():

		# store data
		temp_before_part_id = []
		temp_before_part = []
		temp_after_part_id = []
		temp_after_part = []

		for time_step in range(len(all_part_list)):

			## before region
			# find particles in region
			mp_input_PinR_before = [(part_id, part_i, wall_perform_region[wall_id][0]) for part_id, part_i in enumerate(all_part_list[time_step])]
			part_id_in_before_region_t = pool_PinR.map(part_in_region, mp_input_PinR_before)
			part_id_in_before_region = [part_id for part_id in part_id_in_before_region_t if part_id != None]
			part_in_before_region = [part_i for part_id, part_i in enumerate(all_part_list[time_step]) if part_id in part_id_in_before_region]

			if len(part_in_before_region) > 0:
				temp_before_part_id.append(deepcopy(part_id_in_before_region))
				temp_before_part.append(deepcopy(part_in_before_region))
			else:
				temp_before_part_id.append(None)
				temp_before_part.append(None)

			## after region
			# find particles in region
			mp_input_PinR_after = [(part_id, part_i, wall_perform_region[wall_id][1]) for part_id, part_i in enumerate(all_part_list[time_step])]
			part_id_in_after_region_t = pool_PinR.map(part_in_region, mp_input_PinR_after)
			part_id_in_after_region = [part_id for part_id in part_id_in_after_region_t if part_id != None]
			part_in_after_region = [part_i for part_id, part_i in enumerate(all_part_list[time_step]) if part_id in part_id_in_after_region]

			if len(part_in_after_region) > 0:
				temp_after_part_id.append(deepcopy(part_id_in_after_region))
				temp_after_part.append(deepcopy(part_in_after_region))
			else:
				temp_after_part_id.append(None)
				temp_after_part.append(None)

		# store data
		wall_perform_time[wall_id] = [deepcopy(temp_before_part_id), deepcopy(temp_before_part), deepcopy(temp_after_part_id), deepcopy(temp_after_part)]

		del temp_before_part_id
		del temp_before_part
		del temp_after_part_id
		del temp_after_part

	# stop multiprocessing
	pool_PinR.close()
	pool_PinR.join()

	#############################################
	## barrier performance 
	#############################################
	wall_performance = {}	# key = wall_id, value = [attenuation ratio, trap ratio]
	for wall_id in wall_perform_region.keys():

		part_before_list = wall_perform_time[wall_id][1]
		
		part_id_after_list = wall_perform_time[wall_id][2]
		part_after_list = wall_perform_time[wall_id][3]

		#######################
		## before region
		#######################
		# compute the average velocity and volume sum since the max_part_N_before
		part_av_u_before_list = [np.mean([part_bi.ui for part_bi in part_b_list]) if part_b_list != None else 0.0 for part_b_list in part_before_list]
		part_sum_V_before_list = [sum([part_bi.Vi for part_bi in part_b_list]) if part_b_list != None else 0.0 for part_b_list in part_before_list]

		# maximum av_U and maximum sum_V
		max_part_av_u_before = max(part_av_u_before_list)
		max_part_sum_V_before = max(part_sum_V_before_list)

		#######################
		## after region
		#######################
		# unique particle numbers that passed through the open type wall
		overall_part_id_after_list = [[part_ai_id for part_ai_id in part_a_id_list] for part_a_id_list in part_id_after_list if part_a_id_list != None]
		overall_part_after_list = [[part_ai for part_ai in part_a_list] for part_a_list in part_after_list if part_a_list != None]
		
		if len(overall_part_after_list) > 0:

			unique_part_id_after_list1 = np.unique(overall_part_id_after_list).tolist()
			unique_part_id_after_list2 = []
			for uni_part_id_list in unique_part_id_after_list1:
				for uni_part_id in uni_part_id_list:
					unique_part_id_after_list2.append(uni_part_id)
			unique_part_id_after_list = np.unique(unique_part_id_after_list2).tolist()

			# store speed and volume of the first instances of passing particles
			part_u_after_list = []
			part_V_after_list = []
			for part_a_id_list, part_a_list in zip(overall_part_id_after_list, overall_part_after_list):
				for part_a_id, part_a in zip(part_a_id_list, part_a_list): 
					if part_a_id in unique_part_id_after_list:
						unique_part_id_after_list.remove(part_a_id)
						part_u_after_list.append(part_a.ui)
						part_V_after_list.append(part_a.Vi)

			# maximum av_U and maximum sum_V
			part_av_u_after = np.mean(part_u_after_list)
			part_sum_V_after = sum(part_V_after_list)
		
		else:
			part_av_u_after = 0
			part_sum_V_after = 0

		#######################
		## performance
		#######################
		# performance = [AR, TR] = [(av_U_after/av_U_before), (V_after/V_before)]

		if max_part_av_u_before == 0: 
			speed_reduction = None
		else:
			speed_reduction = (part_av_u_after/max_part_av_u_before)

		if max_part_sum_V_before == 0: 
			volume_reduction = None
		else:
			volume_reduction = (part_sum_V_after/max_part_sum_V_before)

		wall_performance[wall_id] = [speed_reduction, speed_reduction]

	return wall_performance
		

#################################################################################################################
### Optimal closed barrier location selection
#################################################################################################################

# find debris flow parameters - Volume (V), Impact pressure (P), Distance-from-road (D)
def VPD_parameters(flowpath_dfl, dfl_candidate, road_xy_list, dp=4):

	# store dfl VPD data into dictionary
	dfl_V = {}
	dfl_P = {}
	dfl_D = {}
	for dfl_i in dfl_candidate:

		# debris-flow cluster data
		dfl_cluster_class_list = flowpath_dfl[dfl_i][2]

		# volume and impact pressure from cumulation of cluster parameters
		temp_V = []
		temp_P = []
		for cluster_i in dfl_cluster_class_list:
			temp_V.append(cluster_i.Vc)
			temp_P.append(cluster_i.Pc)

		dfl_V[dfl_i] = sum(temp_V)  # sum value
		dfl_P[dfl_i] = sum(temp_P)	# sum value

		# distance-from-road from road
		xr1 = road_xy_list[0][0]
		yr1 = road_xy_list[0][1]
		xr2 = road_xy_list[1][0]
		yr2 = road_xy_list[1][1]
		xc = flowpath_dfl[dfl_i][1][0]
		yc = flowpath_dfl[dfl_i][1][1]
		denominator = np.sqrt((yr1-yr2)**2 + (xr1-xr2)**2)
		numerator = (yr2-yr1)*xc - (xr2-xr1)*yc + xr2*yr1 - xr1*yr2
		dfl_D[dfl_i] = abs(numerator)/denominator

	# compute maximum data
	max_VPD = [max(dfl_V.values()), max(dfl_P.values()), max(dfl_D.values())]

	# compute minimum data
	# min_VPD = [min(dfl_V.values()), min(dfl_P.values()), min(dfl_D.values())] 
	min_VPD = [0, 0, 0]

	del dfl_V
	del dfl_P
	del temp_P
	del temp_V

	return max_VPD, min_VPD, dfl_D

###########################################################################
## optimal closed barrier location selection - children function
###########################################################################

# check whether the set of closed-type barriers successfully mitigate
def cost_closed_barrier_location(dfl_barrier_set, flowpath_link, flowpath_dfl, max_VPD, min_VPD, dfl_D, opt_weight, dp=4): 
	'''
	for closed-type barrier mitigation, assume mitigation successful 
	if the dfl_candidates appears on all the flowpath_link
	'''

	## store data
	source_number = list(flowpath_link.keys())
	all_mitigated_flowpath = []		# all successfully mitigated flowpath
	cID_mitigated_dfl = {}			# key = cID, value = [dfl_list]
	dfl_mitigated_cID = {}			# key = dfl, value = [cID_list]
	dfl_mitigated_flowpath = {}		# key = dfl, value = [[cID_list], [cluster_class], [V, P, D] [normalized V, P, D]
	cost = 0

	# sort dfl so that the smaller dfl number (closer to source)
	# is blocking the debris-flow first
	dfl_barrier_set.sort()

	## mitigation check
	for cID in source_number:
		temp_list = []
		temp_idx_list = []

		for dfl_c in dfl_barrier_set: 
			if dfl_c in flowpath_link[cID][0]: 
				all_mitigated_flowpath.append(cID)
				temp_list.append(dfl_c)
				temp_idx = flowpath_link[cID][0].index(dfl_c)
				temp_idx_list.append(temp_idx)

		cID_mitigated_dfl[cID] = [temp_list[:], temp_idx_list[:]]

	all_mitigated_flowpath.sort()  # sort blocked flowpath list	

	# print()
	# print(source_number, all_mitigated_flowpath, cID_mitigated_dfl)

	## function output depending on the mitigation check
	# not successful mitigation
	if np.unique(all_mitigated_flowpath).tolist() != source_number:
		# mitigation_bool = False, cost = 1 (max cost value), dfl_mitigated_flowpath = not computed
		return False, 100, None

	else:	
		## parameter of debris-flow on each dfl
		# dfl_mitigated_flowpath = {}	# key = dfl, value = [[cID_list], [cluster_class], [V, P, D] [normalized V, P, D]
		# flowpath_dfl: key = dfl id, value = [(0) dfl_type, (1) (x,y), (2) [cluster class]]
		# cluster_class = clusterID, particles, time, row, col, xc, yc, zc, sc, Vc, uc, hc, predecessor, Dc, Pc, ll, merged, dfl_type, dfl_id

		# take only the single dfl per flowpath, one that blocks first
		for cID_j in cID_mitigated_dfl.keys():
			if len(cID_mitigated_dfl[cID_j][0]) > 1:
				min_time = min(cID_mitigated_dfl[cID_j][1])
				min_time_idx = cID_mitigated_dfl[cID_j][1].index(min_time)
				min_time_dfl = cID_mitigated_dfl[cID_j][0][min_time_idx]
				cID_mitigated_dfl[cID_j] = [[min_time_dfl], [min_time]]

		# construct which cluster dfl blocked 
		dfl_barrier_set_final = []
		for cID_k in cID_mitigated_dfl.keys():
			for dfl_f in cID_mitigated_dfl[cID_k][0]:
				if dfl_f not in dfl_barrier_set_final:
					dfl_barrier_set_final.append(dfl_f)

				if dfl_f not in dfl_mitigated_cID.keys():
					dfl_mitigated_cID[dfl_f] = [cID_k]
				else:
					dfl_mitigated_cID[dfl_f].append(cID_k)
		dfl_barrier_set_final.sort()

		total_Vol_norm = 0
		total_Press_norm = 0
		total_Dist_norm = 0
		# for dfl_cc in dfl_barrier_set:
		for dfl_cc in dfl_barrier_set_final:

			# the potential all cluster that passes through the specific dfl
			original_cluster_class_list = flowpath_dfl[dfl_cc][2]

			# find the actual cluster that is being blocked by each dfl
			mitigated_cID_list_i = dfl_mitigated_cID[dfl_cc]
			# try:
			# 	mitigated_cID_list_i = dfl_mitigated_cID[dfl_cc]
			# except:
			# 	dfl_mitigated_flowpath[dfl_cc] = [[], [], [0, 0, 0], [0, 0, 0]]
			# 	continue

			mitigated_cluster_class_list_i = []

			# store parameter
			# volume and impact pressure directly from the cluster parameter
			Vol = 0
			Vol_norm = 0
			Press = 0
			Press_norm = 0
			for cluster_class_i in original_cluster_class_list:
				if cluster_class_i.clusterID in mitigated_cID_list_i:
					mitigated_cluster_class_list_i.append(cluster_class_i)
					Vol += cluster_class_i.Vc
					Vol_norm += round((cluster_class_i.Vc-min_VPD[0])/(max_VPD[0]-min_VPD[0]), dp) 
					Press += cluster_class_i.Pc
					Press_norm += round((cluster_class_i.Pc-min_VPD[1])/(max_VPD[1]-min_VPD[1]), dp) 

			# location is fixed regardless of cluster; therefore, use computed dfl_D
			if Vol > 0 and Press > 0:
				Dist = dfl_D[dfl_cc]
				Dist_norm = round((Dist-min_VPD[2])/(max_VPD[2]-min_VPD[2]), dp) 
			else:
				Dist = 0
				Dist_norm = 0

			# store mitigation dfl dictionary
			dfl_mitigated_flowpath[dfl_cc] = [mitigated_cID_list_i, mitigated_cluster_class_list_i, [Vol, Press, Dist], [Vol_norm, Press_norm, Dist_norm]]

			# summation for computing cost
			total_Vol_norm += Vol_norm
			total_Press_norm += Press_norm
			total_Dist_norm += Dist_norm

		## compute the optimization cost
		# cost = weight*Parameters = w1*Volume + w2*(Impact Pressure) + w3*(Distance-from-road) + w4*(N = number of barriers)
		cost = opt_weight[0]*total_Vol_norm + opt_weight[1]*total_Press_norm + opt_weight[2]*total_Dist_norm + opt_weight[3]*len(dfl_barrier_set)

		return True, cost, dfl_mitigated_flowpath


# exclude certain dfl candidates depending on set of criteria
def generate_avoid_dfl_closed(flowpath_link, min_dfl_ordinal=0.3):
	# removes any dfl that are too close to the source location

	dfl_candidate_new = []

	for cID in flowpath_link.keys():

		# network data
		dfl_links = flowpath_link[cID][0]
		dfl_types = flowpath_link[cID][1]

		# complete merging
		try: 		# find first instance of complete merging
			first_dfl_type_2_idx = dfl_types.index(2)	
		except:		# dfl_type 2 does not exist
			first_dfl_type_2_idx = len(dfl_types)

		# spatial only merging
		try: 		# find first instance of spatial only merging
			first_dfl_type_3_idx = dfl_types.index(3)	
		except:		# dfl_type 3 does not exist
			first_dfl_type_3_idx = len(dfl_types)

		# take the dfl at 30% from source to the terminus
		ordinal_idx = int(len(dfl_types)*min_dfl_ordinal)
					
		# find the starting dfl index
		start_idx = min([first_dfl_type_2_idx, first_dfl_type_3_idx, ordinal_idx])

		# store new dfl candidate list
		for new_dfl in dfl_links[start_idx:]:
			if new_dfl not in dfl_candidate_new:
				dfl_candidate_new.append(new_dfl)

	return dfl_candidate_new


# assign new set of dfl_barriers to check
def assign_dfl_barrier_sets(iteration_no, b_no, old_dfl_barrier, dfl_candidate, flowpath_link, num_dfl_start, checked_dfl_set, find_iter_limit):

	# create list of places to potentially place the barrier
	cluster_id_list = [i for i in range(num_dfl_start)]  # also the cluster id list
	# find_iter_cID = 0

	# if first time, randomly select a dfl
	if iteration_no == 0:

		dfl_barrier_set = []
	
		# select the flowpath to set barrier
		if b_no <= len(cluster_id_list):
			cluster_chosen = np.random.choice(cluster_id_list, b_no, replace=False)
		else:
			cluster_chosen = np.random.choice(cluster_id_list, b_no)

		for cID in cluster_chosen:
			while True:
				dfl_selected = np.random.choice(flowpath_link[cID][0], 1).tolist()[0]
				if dfl_selected not in dfl_barrier_set and dfl_selected in dfl_candidate:
					dfl_barrier_set.append(dfl_selected)
					dfl_barrier_set.sort()
					break
	
	# if not first time, randomly select one of the dfl and replace
	# cluster ID -> dfl number to place -> replacing dfl number
	else:
		iter_replace_cID = 0
		while iter_replace_cID < 5:
			cluster_chosen = np.random.choice(cluster_id_list, 1)
			cluster_chosen = cluster_chosen.tolist()[0]

			if b_no == 1:
				dfl_barrier_set = [np.random.choice(dfl_candidate, 1).tolist()[0]]

			else:
				dfl_barrier_set = old_dfl_barrier[:]

				iter_replace_idx = 0
				replace_dfl_bool = True
				while replace_dfl_bool:

					dfl_replace_chosen = np.random.choice(dfl_barrier_set, 1)
					dfl_replace_chosen_index = dfl_barrier_set.index(dfl_replace_chosen.tolist()[0])

					while True:

						if iter_replace_idx > find_iter_limit:
							replace_dfl_bool = False
							break

						dfl_selected = np.random.choice(flowpath_link[cluster_chosen][0], 1).tolist()[0]
						if dfl_selected not in dfl_barrier_set and dfl_selected in dfl_candidate:
							dfl_barrier_set[dfl_replace_chosen_index] = dfl_selected
							dfl_barrier_set.sort()
							replace_dfl_bool = False
							break
						else:
							iter_replace_idx += 1

			if dfl_barrier_set not in checked_dfl_set:
				break
			else:
				iter_replace_cID += 1

	return dfl_barrier_set


###########################################################################
## optimal combined barrier location selection - children function
###########################################################################

# check the vulnerability of debris-flow based on Kang and Kim (2014)
def vulnerability_index_u(velocity, RC=True):
	if RC:
		# VI = 1 - np.exp(-0.064*(velocity**1.625))  	# Kang and Kim 2014
		VI = 0.71*(1 - np.exp(-0.013*(velocity**2.89)))	# 2021 
	else:
		# VI = 1 - np.exp(-0.018*(velocity**4.075))	# Kang and Kim 2014
		VI = 1.01*(1 - np.exp(-0.148*(velocity**1.69)))	# 2021 
	return round(VI,2)

# check the vulnerability of debris-flow based on Kang and Kim (2014)
def vulnerability_index_P(pressure, RC=True):
	if RC:
		# VI = 1 - np.exp(-0.0167*(pressure**0.917))  	# Kang and Kim 2014
		VI = 0.71*(1 - np.exp(-0.0008*(pressure**1.82)))	# 2021 
	else:
		# VI = 1 - np.exp(-0.004*(pressure**1.812))	# Kang and Kim 2014
		VI = 1.00*(1 - np.exp(-0.0272*(pressure**1.10)))	# 2021 
	return round(VI,2)


# check whether the set of closed-type barriers successfully mitigate
def cost_combined_barrier_location_v1_0(dfl_barrier_set, flowpath_dfl, cID_uV_dict, dfl_uV_dict, open_performance, max_VPD, min_VPD, dfl_D, alpha, material, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, opt_weight, dp=4): 
	'''
	for closed-type barrier mitigation, assume mitigation successful 
	if the dfl_candidates appears on all the flowpath_link
	already successful mitigation requirement met when being assigned
	'''
	
	# cID_uV_dict = {}   # {dfl_id: [[cluster_id, [speed, volume]], ...]}
	# dfl_uV_dict = {}   # {dfl_id: [impact speed, impact volume]}	# impact speed = average(cluster speed); impact volume = sum(cluster volume)
	# flowpath_dfl = {} 	# key = dfl id, value = [(0) dfl_type, (1) (x,y), (2) [cluster class]]

	## store data
	dfl_mitigated_flowpath = {}		# key = dfl, value = [barrier_performance, [cID_list], [V, P, D] [normalized V, P, D]

	## compute total cost variables
	total_Vol_norm = 0
	total_Press_norm = 0
	total_Dist_norm = 0
	total_barrier_number = 0
	complete_dfl_barrier_set = dfl_barrier_set[0] + dfl_barrier_set[1]
	for dfl_cc in complete_dfl_barrier_set:

		# extract impact volume directly from dfl_uV_dict
		Vol = dfl_uV_dict[dfl_cc][1]
		Vol_norm = round((Vol-min_VPD[0])/(max_VPD[0]-min_VPD[0]), dp) 

		## impact pressure directly from the cluster parameter
		# material properties
		dfl_xy = flowpath_dfl[dfl_cc][1]
		f, Es, density, phi = local_mat_v2_0(material, dfl_xy[0], dfl_xy[1], MAT, gridUniqueX, gridUniqueY, deltaX, deltaY)

		# P = alpha*density*(speed^2)
		Press = alpha*(density/1000)*((dfl_uV_dict[dfl_cc][0])**2)  # kPa
		Press_norm = round((Press-min_VPD[1])/(max_VPD[1]-min_VPD[1]), dp) 

		# location is fixed regardless of cluster; therefore, use computed dfl_D
		Dist = dfl_D[dfl_cc]
		Dist_norm = round((Dist-min_VPD[2])/(max_VPD[2]-min_VPD[2]), dp) 

		# all the clusters being blocked/affected
		affected_cID_list_i = [result_list[0] for result_list in cID_uV_dict[dfl_cc]]

		# performance
		if dfl_cc in dfl_barrier_set[0]:
			barrier_performance = open_performance[:]
		elif dfl_cc in dfl_barrier_set[1]:
			barrier_performance = [1.0, 1.0]

		# store mitigation dfl dictionary
		dfl_mitigated_flowpath[dfl_cc] = [deepcopy(barrier_performance), deepcopy(affected_cID_list_i), [Vol, Press, Dist], [Vol_norm, Press_norm, Dist_norm]]

		# summation for computing cost
		total_Vol_norm += Vol_norm
		total_Press_norm += Press_norm
		total_Dist_norm += Dist_norm
		total_barrier_number += 1

	## compute the optimization cost
	# cost = weight*Parameters = w1*Volume + w2*(Impact Pressure) + w3*(Distance-from-road) + w4*(N-open = number of open-type barriers) + w5*(N-closed = number of closed-type barriers)
	cost = opt_weight[0]*total_Vol_norm + opt_weight[1]*total_Press_norm + opt_weight[2]*total_Dist_norm + opt_weight[3]*len(dfl_barrier_set[0]) + opt_weight[4]*len(dfl_barrier_set[1])

	del total_Vol_norm
	del total_Press_norm
	del total_Dist_norm 
	del total_barrier_number
	del complete_dfl_barrier_set 

	return cost, dfl_mitigated_flowpath


# exclude certain dfl candidates depending on set of criteria
def generate_avoid_dfl_combined(flowpath_link, min_dfl_ordinal=0.3):
	# removes any dfl that are too close to the source location

	dfl_candidate_new = []

	for cID in flowpath_link.keys():

		# network data
		dfl_links = flowpath_link[cID][0]
		dfl_types = flowpath_link[cID][1]

		# complete merging
		try: 		# find first instance of complete merging
			first_dfl_type_2_idx = dfl_types.index(2)	
		except:		# dfl_type 2 does not exist
			first_dfl_type_2_idx = len(dfl_types)

		# spatial only merging
		try: 		# find first instance of spatial only merging
			first_dfl_type_3_idx = dfl_types.index(3)	
		except:		# dfl_type 3 does not exist
			first_dfl_type_3_idx = len(dfl_types)

		# take the dfl at 30% from source to the terminus
		ordinal_idx = int(len(dfl_types)*min_dfl_ordinal)
					
		# find the starting dfl index
		start_idx = min([first_dfl_type_2_idx, first_dfl_type_3_idx, ordinal_idx])

		# find the ending dfl index
		if dfl_types[-1] == 4:
			end_idx = len(dfl_types)-1
		else:
			end_idx = len(dfl_types)

		# store new dfl candidate list - starting to dfl just before terminus
		for new_dfl in dfl_links[start_idx:end_idx]:
			if new_dfl not in dfl_candidate_new:
				dfl_candidate_new.append(new_dfl)

	return dfl_candidate_new


# debris-flow character at the end of the flowpath network terminus or any terminating point
def compute_debris_flow_open_terminus_SPEC_v1_0(flowpath_link, flowpath_dfl, network_data, open_dfl_list, open_performance, cell_size, fb_0, g, material, MAT, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, entrainment_model, interp_method, min_uV, Es_theta_var):

	# network_data = [num_dfl_start, source_dfl_list, len(terminus_dfl_list), terminus_dfl_list]
	# open_barrier_cluster = {open type dfl : [list of affected cluster id]}
	# open_dfl_list = [list of dfl_id where open-type barrier is placed]
	# cluster_dfl_barrier_set = {cluster_id : [[open-type set], [closed-type set]]}
	# flowpath_link = {}	# key = cluster id, value = [(0) [linking dfl_ids], (1) [dfl_type]]
	# flowpath_dfl = {} 	# key = dfl id, value = [(0) dfl_type, (1) (x,y), (2) [cluster class]]
	# cID_first_open_dfl_data = 
	# 		{cID: [(0)[start_dfl_id, terminus_dfl_id], (1)[u*SR, V*SR at the first dfl that appeared along the flowpath],
	#		 (2)[dfl_list(first_open/source -> terminus)], (3)[cID_list(first_open/source -> terminus)], (4)[dfl_xyz_list(first_open/source -> terminus)]]}
	# 	if dfl_type == [2 (merge), 3 (spatial merge), 4 (terminus with multiple clusters)], cID_list = [cluster_id involved]

	open_dfl_list.sort()

	## compute the initial velocity and volume after considering the open_performance
	cID_first_open_dfl_data = {}  
	cID_uV_dict = {}   # {dfl_id: [[cluster_id, [speed, volume]], ...]}
	dfl_uV_dict = {}   # {dfl_id: [impact speed, impact volume]}	# impact speed = average(cluster speed); impact volume = sum(cluster volume)
	for cID in range(network_data[0]):

		dfl_present_list = []
		dfl_index_list = []
		for dfl_open in open_dfl_list:
			if dfl_open in flowpath_link[cID][0]:
				dfl_present_list.append(dfl_open)
				link_idx = flowpath_link[cID][0].index(dfl_open)
				dfl_index_list.append(link_idx)
		
		if len(dfl_present_list) == 0:
			first_open_dfl = None
		elif len(dfl_present_list) == 1: 
			first_open_dfl = dfl_present_list[0]
		elif len(dfl_present_list) > 1: 
			min_dfl_idx = min(dfl_index_list)
			first_open_dfl = flowpath_link[cID][0][min_dfl_idx]
		
		## find the starting dfl for each flowpath 
		# add any other open-type dfl present
		if first_open_dfl != None:

			dfl_open_cluster_list = flowpath_dfl[first_open_dfl][2]
			dfl_open_type = flowpath_dfl[first_open_dfl][0]

			impact_u = 0
			impact_V = 0
			count = 0
			for cluster_o in dfl_open_cluster_list:
				# for spatial merged cluster, separate
				if dfl_open_type in [3,4]: 
					if cluster_o.clusterID == cID:
						impact_u += cluster_o.uc
						impact_V += cluster_o.Vc
						count += 1
				
				# if completely merged, separate by computing SPEC-debris
				elif dfl_open_type == 2: 

					dfl_idx = flowpath_link[cID][0].index(first_open_dfl)
					pre_dfl = flowpath_link[cID][0][dfl_idx-1]

					cluster_cID = [cluster_oo for cluster_oo in flowpath_dfl[pre_dfl][2] if cluster_oo.clusterID == cID][0]

					# xy coordinate
					cur_dfl_xy = flowpath_dfl[first_open_dfl][1]
					pre_dfl_xy = flowpath_dfl[pre_dfl][1]
					
					# find local region 
					local_cur_xy, local_cur_z = local_cell_v3_0(cell_size, cur_dfl_xy[0], cur_dfl_xy[1], DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
					local_pre_xy, local_pre_z = local_cell_v3_0(cell_size, pre_dfl_xy[0], pre_dfl_xy[1], DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
					
					# find elevation z coorindate
					cur_dfl_z = compute_Z_v3_0([cur_dfl_xy[0], cur_dfl_xy[1]], local_cur_xy, local_cur_z, interp_method)
					pre_dfl_z = compute_Z_v3_0([pre_dfl_xy[0], pre_dfl_xy[1]], local_pre_xy, local_pre_z, interp_method)

					# material properties
					f, Es, density, phi = local_mat_v2_0(material, local_pre_xy[0], local_pre_xy[1], MAT, gridUniqueX, gridUniqueY, deltaX, deltaY)

					# compute volume and velocity
					if entrainment_model == 'Hungr':
						# uf, Vf = SEC_Hungr_v5_0([pre_dfl_xy[0], pre_dfl_xy[1], pre_dfl_z], [cur_dfl_xy[0], cur_dfl_xy[1], cur_dfl_z], cluster_cID.uc, cluster_cID.Vc, cluster_cID.hc, f[0], f[1], Es, g, None, None, theta_var=Es_theta_var)
						uf, Vf = SEC_Hungr_v5_0([pre_dfl_xy[0], pre_dfl_xy[1], pre_dfl_z], [cur_dfl_xy[0], cur_dfl_xy[1], cur_dfl_z], cluster_cID.uc, cluster_cID.Vc, None, f[0], f[1], Es, g, None, None, theta_var=Es_theta_var)
						# uf, Vf = SEC_Hungr_v6_0([pre_dfl_xy[0], pre_dfl_xy[1], pre_dfl_z], [cur_dfl_xy[0], cur_dfl_xy[1], cur_dfl_z], cluster_cID.uc, cluster_cID.Vc, cluster_cID.hc, f[0], f[1], Es, g, None, density, theta_var=Es_theta_var)
					elif entrainment_model == 'Er':
						uf, Vf = SEC_Er_v5_0([pre_dfl_xy[0], pre_dfl_xy[1], pre_dfl_z], [cur_dfl_xy[0], cur_dfl_xy[1], cur_dfl_z], cluster_cID.uc, cluster_cID.Vc, None, f[0], f[1], Es, g, None, None, theta_var=Es_theta_var)
						# uf, Vf = SEC_Er_v6_0([pre_dfl_xy[0], pre_dfl_xy[1], pre_dfl_z], [cur_dfl_xy[0], cur_dfl_xy[1], cur_dfl_z], cluster_cID.uc, cluster_cID.Vc, cluster_cID.hc, f[0], f[1], Es, g, None, density, theta_var=Es_theta_var)

					impact_u += uf
					impact_V += Vf
					count += 1

				else:
					impact_u += cluster_o.uc
					impact_V += cluster_o.Vc
					count += 1

			# cID_uV_dict
			if first_open_dfl in cID_uV_dict.keys():
				temp_result_list = cID_uV_dict[first_open_dfl][:]
				temp_result_list.append([cID, [open_performance[0]*(impact_u/count), open_performance[1]*impact_V]])
				cID_uV_dict[first_open_dfl] = temp_result_list[:]
				del temp_result_list
			else:
				cID_uV_dict[first_open_dfl] = [[cID, [open_performance[0]*(impact_u/count), open_performance[1]*impact_V]]]
			
			# dfl_uV_dict
			if first_open_dfl in dfl_uV_dict.keys():
				temp_impact_list = dfl_uV_dict[first_open_dfl][:]
				# impact speed = average(cluster speed) 
				temp_impact_list[0] = (temp_impact_list[0]*len(cID_uV_dict[first_open_dfl]) + open_performance[0]*(impact_u/count))/(len(cID_uV_dict[first_open_dfl])+1)
				# impact volume = sum(cluster volume)
				temp_impact_list[1] = temp_impact_list[1] + open_performance[1]*impact_V
				dfl_uV_dict[first_open_dfl] = temp_impact_list[:]
				del temp_impact_list
			else:
				dfl_uV_dict[first_open_dfl] = [open_performance[0]*(impact_u/count), open_performance[1]*impact_V]

		# find first open_type barrier existed in the flowpath
		if first_open_dfl == None:
			# first_open_dfl = None -> starts from the source location 
			source_dfl = flowpath_link[cID][0][0]
			source_cluster = flowpath_dfl[source_dfl][2][0]
			cID_first_open_dfl_data[cID] = [[None], [source_cluster.uc, source_cluster.Vc]]
				
		else:
			cID_uV_list = cID_uV_dict[first_open_dfl]
			for nn in range(len(cID_uV_list)):
				if cID_uV_list[nn][0] == cID:
					cID_first_open_dfl_data[cID] = [[first_open_dfl], cID_uV_list[nn][1][:]]
		
		## find, which path is taken to reach a terminus 
		# if the final dfl is terminus
		if flowpath_link[cID][1][-1] == 4:
			ending_dfl_id = flowpath_link[cID][0][-1]

			# starts from source
			if cID_first_open_dfl_data[cID][0][0] == None: 
				start_dfl_index = 0
			# start from the first instance of open-type barrier
			else:
				start_dfl_index = flowpath_link[cID][0].index(cID_first_open_dfl_data[cID][0][0])
			start_to_end_dfl_list = flowpath_link[cID][0][start_dfl_index:]

		# if the final dfl ends with merging
		elif flowpath_link[cID][1][-1] == 2:
			
			# the dfl from other cluster
			merging_dfl = flowpath_link[cID][0][-1]
			other_cID_list = [loop for loop in range(network_data[0]) if loop != cID]
			other_cID_f = None
			junction_cID_list = []
			while True:
				other_cID_contain = [other_cID for other_cID in other_cID_list if merging_dfl in flowpath_link[other_cID][0]]
				if len(other_cID_contain) == 1:
					if flowpath_link[other_cID_contain[0]][1][-1] == 4:
						other_cID_f = other_cID_contain[0]
				elif len(other_cID_contain) > 1:
					other_cID_f_list = []
					for check_terminus_cID in other_cID_contain:
						if flowpath_link[check_terminus_cID][1][-1] == 4:
							other_cID_f_list.append(check_terminus_cID)
					
					if len(other_cID_f_list) == 1: 
						other_cID_f = other_cID_f_list[0]

				if other_cID_f == None:
					junction_cID_list.append(other_cID_f)
				else:
					break
			
			# ending dfl id
			ending_dfl_id = flowpath_link[other_cID_f][0][-1]

			# series of dfl from start to ending
			start_to_end_dfl_list = []
			cID_temp_list = [cID] + junction_cID_list + [other_cID_f]
			for loopN, cID_temp in enumerate([cID, other_cID_f]):

				if cID_temp == cID:
					# starts from source			
					if cID_first_open_dfl_data[cID_temp][0][0] == None: 
						start_dfl_index = 0
					# start from the first instance of open-type barrier
					else:
						start_dfl_index = flowpath_link[cID_temp][0].index(cID_first_open_dfl_data[cID_temp][0][0])
				else:
					start_dfl_index = flowpath_link[cID_temp][0].index(flowpath_link[cID_temp_list[loopN-1]][0][-1])

				for dfl_temp in flowpath_link[cID_temp][0][start_dfl_index:]:
					if dfl_temp not in start_to_end_dfl_list:
						start_to_end_dfl_list.append(dfl_temp)

		## find involved cluster numbers
		start_to_end_dfl_cID_list = []
		start_to_end_dfl_xyz_list = []
		for temp_dfl in start_to_end_dfl_list:
			## find involved cluster number 
			# all but merging has cluster to find
			if flowpath_dfl[temp_dfl][0] in [0,1,3,4]:
				temp_cID_list = [cluster_temp.clusterID for cluster_temp in flowpath_dfl[temp_dfl][2]]
			# for merging dfl_type, find using dfl_id number present in flowpath_link
			elif flowpath_dfl[temp_dfl][0] == 2:
				temp_cID_list = [check_cID for check_cID in range(network_data[0]) if temp_dfl in flowpath_link[check_cID][0]]
			start_to_end_dfl_cID_list.append(temp_cID_list[:])
			del temp_cID_list

			## xyz coordinates
			# xy coordinate
			dfl_xy = flowpath_dfl[temp_dfl][1]
			# find elevation z coorindate
			local_xy_f, local_z_f = local_cell_v3_0(cell_size, dfl_xy[0], dfl_xy[1], DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
			dfl_z = compute_Z_v3_0([dfl_xy[0], dfl_xy[1]], local_xy_f, local_z_f, interp_method)
			# store
			start_to_end_dfl_xyz_list.append([dfl_xy[0], dfl_xy[1], dfl_z])

		# store data into dictionary
		temp_list = cID_first_open_dfl_data[cID]
		temp_list[0].append(ending_dfl_id)  # [starting_dfl_id, terminus_dfl_id]
		temp_list.append(start_to_end_dfl_list)
		temp_list.append(start_to_end_dfl_cID_list)
		temp_list.append(start_to_end_dfl_xyz_list)
		cID_first_open_dfl_data[cID] = temp_list[:]
		del temp_list

	# check whether cID_first_open_dfl_data computed correctly
	computed_terminus_dfl_list = [cID_first_open_dfl_data[cIDc][0][1] for cIDc in cID_first_open_dfl_data.keys()]
	if np.unique(computed_terminus_dfl_list).tolist() != sorted(network_data[3]):
		print(cID_first_open_dfl_data)
		assert False, "computation error occured at cID_first_open_dfl_data"

	
	## compute the debris-flow parameters along the flowpath
	# compute for each debris-flow cluster - speed and volume at open_barrier and terminus
	# cID_uV_dict = {}   # {dfl_id: [[cluster_id, [speed, volume]], ...]}
	# dfl_uV_dict = {}   # {dfl_id: [impact speed, impact volume]}	# impact speed = average(cluster speed); impact volume = sum(cluster volume)
	for cID in cID_first_open_dfl_data.keys():
		dfl_uV = [cID_first_open_dfl_data[cID][1]]
		flowpath_dfl_list = cID_first_open_dfl_data[cID][2]
		# flowpath_dfl_type_list = cID_first_open_dfl_data[cID][3]
		dfl_xyz = cID_first_open_dfl_data[cID][4]

		# iterative compute the velocity and volume along the dfl 
		loop = 0
		while loop < len(dfl_xyz)-1:

			dfl_id_next = flowpath_dfl_list[loop+1]

			# material properties
			f, Es, density, phi = local_mat_v2_0(material, dfl_xyz[loop][0], dfl_xyz[loop][1], MAT, gridUniqueX, gridUniqueY, deltaX, deltaY)

			# compute volume and velocity
			if entrainment_model == 'Hungr':
				# uf, Vf = SEC_Hungr_v5_0(dfl_xyz[loop], dfl_xyz[loop+1], dfl_uV[loop][0], dfl_uV[loop][1], None, f[0], f[1], Es, g, None, None, theta_var=Es_theta_var, fb_0=fb_0)
				uf, Vf = SEC_Hungr_v5_0(dfl_xyz[loop], dfl_xyz[loop+1], dfl_uV[loop][0], dfl_uV[loop][1], None, None, None, Es, g, None, None, theta_var=Es_theta_var, fb_0=fb_0)
				# uf, Vf = SEC_Hungr_v6_0(dfl_xyz[loop], dfl_xyz[loop+1], dfl_uV[loop][0], dfl_uV[loop][1], None, f[0], f[1], Es, g, None, density, theta_var=Es_theta_var, fb_0=fb_0)
			elif entrainment_model == 'Er':
				# uf, Vf = SEC_Er_v5_0(dfl_xyz[loop], dfl_xyz[loop+1], dfl_uV[loop][0], dfl_uV[loop][1], None, f[0], f[1], Es, g, None,  None, theta_var=Es_theta_var, fb_0=fb_0)
				uf, Vf = SEC_Er_v5_0(dfl_xyz[loop], dfl_xyz[loop+1], dfl_uV[loop][0], dfl_uV[loop][1], None, None, None, Es, g, None,  None, theta_var=Es_theta_var, fb_0=fb_0)
				# uf, Vf = SEC_Er_v6_0(dfl_xyz[loop], dfl_xyz[loop+1], dfl_uV[loop][0], dfl_uV[loop][1], None, f[0], f[1], Es, g, None,  density, theta_var=Es_theta_var, fb_0=fb_0)


			# if computed speed and volume is below the critical minimum value
			if uf <= min_uV[0] or Vf <= min_uV[1]:
				uf = max(min_uV[0], uf)
				Vf = max(min_uV[1], Vf)

			# store computed debris-flow cluster speed and volume
			already_stored_check = False

			# if open type barrier is present
			if dfl_id_next in open_dfl_list:
				uff = open_performance[0]*uf
				Vff = open_performance[1]*Vf
				dfl_uV.append([uff, Vff])
				already_stored_check = True

				# cID_uV_dict
				if dfl_id_next in cID_uV_dict.keys():
					temp_result_list = cID_uV_dict[dfl_id_next][:]
					temp_result_list.append([cID, [uff, Vff]])
					cID_uV_dict[dfl_id_next] = temp_result_list[:]
					del temp_result_list
				else:
					cID_uV_dict[dfl_id_next] = [[cID, [uff, Vff]]]

				# dfl_uV_dict
				if dfl_id_next in dfl_uV_dict.keys():
					temp_impact_list = dfl_uV_dict[dfl_id_next][:]
					# impact speed = average(cluster speed) 
					temp_impact_list[0] = (temp_impact_list[0]*len(cID_uV_dict[dfl_id_next]) + uff)/(len(cID_uV_dict[dfl_id_next])+1)
					# impact volume = sum(cluster volume)
					temp_impact_list[1] = temp_impact_list[1] + Vff
					dfl_uV_dict[dfl_id_next] = temp_impact_list[:]
					del temp_impact_list
				else:
					dfl_uV_dict[dfl_id_next] = [uff, Vff]

			# if it reaches the terminus
			if dfl_id_next in network_data[3]:
				dfl_uV.append([uf, Vf])
				already_stored_check = True

				# cID_uV_dict
				if dfl_id_next in cID_uV_dict.keys():
					temp_result_list = cID_uV_dict[dfl_id_next][:]
					temp_result_list.append([cID, [uf, Vf]])
					cID_uV_dict[dfl_id_next] = temp_result_list[:]
					del temp_result_list
				else:
					cID_uV_dict[dfl_id_next] = [[cID, [uf, Vf]]]
				
				# dfl_uV_dict
				if dfl_id_next in dfl_uV_dict.keys():
					temp_impact_list = dfl_uV_dict[dfl_id_next][:]
					# impact speed = average(cluster speed) 
					temp_impact_list[0] = (temp_impact_list[0]*len(cID_uV_dict[dfl_id_next]) + uf)/(len(cID_uV_dict[dfl_id_next])+1)
					# impact volume = sum(cluster volume)
					temp_impact_list[1] = temp_impact_list[1] + Vf
					dfl_uV_dict[dfl_id_next] = temp_impact_list[:]
					del temp_impact_list
				else:
					dfl_uV_dict[dfl_id_next] = [uf, Vf]

			# all other
			if already_stored_check == False:
				dfl_uV.append([uf, Vf])

			# add loop number to the next iteration number
			loop += 1

	del cID_first_open_dfl_data
	del computed_terminus_dfl_list

	return dfl_uV_dict, cID_uV_dict


# assign new set of dfl_barriers to check
def assign_dfl_combined_barrier_sets_v3_0(iteration_no, b_no, old_dfl_barrier, num_dfl_start, checked_open_dfl_set, find_iter_limit, dfl_candidate, flowpath_link, flowpath_dfl, network_data, open_performance, cell_size, g, material, MAT, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, entrainment_model, interp_method, min_uV, Es_theta_var, fb_0, VI_crit, RC_bool):

	# dfl_barrier_set = [[open-type set], [closed-type set]]
	# open_barrier_cluster = {open type dfl : [cluster id]}
	# cluster_dfl_barrier_set = {cluster_id : [[open-type set], [closed-type set]]}

	# create list of places to potentially place the barrier
	cluster_id_list = [i for i in range(num_dfl_start)]  # also the cluster id list
	# find_iter_cID = 0

	# if first time, randomly select a dfl for placing open-type barriers, then maybe add closed-type if needed
	if iteration_no == 0:

		dfl_open_barrier_set = []
		# cluster_dfl_barrier_set = {}
		
		# select the flowpath to set barrier
		if b_no <= len(cluster_id_list):
			cluster_chosen = np.random.choice(cluster_id_list, b_no, replace=False)
		else:
			cluster_chosen = np.random.choice(cluster_id_list, b_no)

		# select one dfl from each flowpath to place open-type barrier
		for cID in cluster_chosen:

			# select never picked before open-type barrier location
			while True:
				dfl_open_selected = np.random.choice(flowpath_link[cID][0], 1).tolist()[0]
				if dfl_open_selected not in dfl_open_barrier_set and dfl_open_selected in dfl_candidate:
					dfl_open_barrier_set.append(dfl_open_selected)
					dfl_open_barrier_set.sort()
					break
		
		del cluster_chosen
		del dfl_open_selected
	
	# if not first time, randomly select one of the dfl and replace
	# cluster ID -> dfl number to place -> replacing dfl number
	else:
		# replace open-type barrier location
		iter_replace_cID = 0
		while iter_replace_cID < 5:

			cluster_chosen = np.random.choice(cluster_id_list, 1)
			cluster_chosen = cluster_chosen.tolist()[0]

			if b_no == 1:
				dfl_open_barrier_set = [np.random.choice(dfl_candidate, 1).tolist()[0]]

			else:
				dfl_open_barrier_set = old_dfl_barrier[0][:]

				iter_replace_idx = 0
				replace_dfl_bool = True
				while replace_dfl_bool:

					dfl_replace_chosen = np.random.choice(dfl_open_barrier_set, 1)
					dfl_replace_chosen_index = dfl_open_barrier_set.index(dfl_replace_chosen.tolist()[0])

					while True:

						if iter_replace_idx > find_iter_limit:
							replace_dfl_bool = False
							break

						dfl_selected = np.random.choice(flowpath_link[cluster_chosen][0], 1).tolist()[0]
						if dfl_selected not in dfl_open_barrier_set and dfl_selected in dfl_candidate:
							dfl_open_barrier_set[dfl_replace_chosen_index] = dfl_selected
							dfl_open_barrier_set.sort()
							replace_dfl_bool = False
							break
						else:
							iter_replace_idx += 1
				
				del dfl_selected
				del dfl_replace_chosen
				del dfl_replace_chosen_index
				del iter_replace_idx
				del replace_dfl_bool

			if dfl_open_barrier_set not in checked_open_dfl_set:
				break
			else:
				iter_replace_cID += 1

		del cluster_chosen
		del iter_replace_cID


	## check whether closed-type barrier is required
	# compute where debris-flow will terminate due to the open-type barriers
	dfl_uV_dict, cID_uV_dict = compute_debris_flow_open_terminus_SPEC_v1_0(flowpath_link, flowpath_dfl, network_data, dfl_open_barrier_set, open_performance, cell_size, fb_0, g, material, MAT, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, entrainment_model, interp_method, min_uV, Es_theta_var)

	# if the debris-flow reaches until the terminus dfl and vulnerability is above the acceptable level, add closed-type barrier at terminus
	dfl_closed_barrier_set = []
	dfl_terminus_VI = {}  # {terminus_dfl : [velocity, VI]}
	for dfl_t in network_data[3]:
		if dfl_t in dfl_uV_dict.keys():
			velocity = dfl_uV_dict[dfl_t][0]
			VI = vulnerability_index_u(velocity, RC=RC_bool)  # vulnerability index 	
			dfl_terminus_VI[dfl_t] = [velocity, VI]

			if VI > VI_crit:  
				dfl_closed_barrier_set.append(dfl_t)

	if sorted(list(dfl_terminus_VI.keys())) != sorted(network_data[3]):
		print("error at computing dfl_terminus_VI")
		assert False

	# dfl_barrier_set = [[open-type set], [closed-type set]]
	dfl_barrier_set = [deepcopy(dfl_open_barrier_set), deepcopy(dfl_closed_barrier_set)]

	del dfl_open_barrier_set
	del dfl_closed_barrier_set
	del cluster_id_list
		
	return dfl_barrier_set, dfl_uV_dict, cID_uV_dict, dfl_terminus_VI


#################################################################################################################
### convert SPEC-debris cluster class results into JSON format
#################################################################################################################
# import class object into JSON
def cluster_class_2_json_MP(cluster_input):

	cID, cII, cluster_i = cluster_input

	cluster_i_dict = deepcopy(cluster_i.__dict__)  # export class into dictionary

	# remove predecessor, particles and boundary polygon (class type)
	del cluster_i_dict['predecessor']
	del cluster_i_dict['particles']
	del cluster_i_dict['boundary_polygon']

	# convert numpy array into list
	boundary_x, boundary_y = cluster_i_dict['boundary_pt_xy']
	boundary_xy_tuple = (boundary_x.tolist(), boundary_y.tolist())
	cluster_i_dict['boundary_pt_xy'] = deepcopy(boundary_xy_tuple)

	# convert dictionary into JSON string and add to list
	cluster_str = json.dumps(cluster_i_dict)
	
	# return (cID, cII, deepcopy(cluster_i_dict))
	return (cID, cII, cluster_str)

def cluster_class_2_json_MP_v2(cluster_input):
	
	cID, cII, cluster_i = cluster_input

	cluster_i_dict = deepcopy(cluster_i.__dict__)  # export class into dictionary

	# remove predecessor, particles, boundary polygon and boundary points (class type)
	del cluster_i_dict['predecessor']
	del cluster_i_dict['particles']
	del cluster_i_dict['boundary_polygon']
	del cluster_i_dict['boundary_pt_xy']

	# convert dictionary into JSON string and add to list
	cluster_str = json.dumps(cluster_i_dict)
	
	# return (cID, cII, deepcopy(cluster_i_dict))
	return (cID, cII, cluster_str)

#################################################################################################################
### SPEC-debris results plot
#################################################################################################################
###########################################################################
## 2D and 3D plotly interactive map - HTML
###########################################################################
# max particle value (h, u, V, P) at each grid cell
def max_part_data_at_cell_MP(part_data_cell_input):

	part_df, rr, cc = part_data_cell_input

	part_t_at_cell = np.array(part_df[((part_df['i'] == rr) & (part_df['j'] == cc))])

	if len(part_t_at_cell) > 0:
		# 'pID0','cID1','t2','s3','x4','y5','z6','elevation7','u8','ux9','uy10','h11','V12','P13','i14','j15'
		cell_part_max_eli = max(part_t_at_cell[:,7])
		cell_part_max_ui = max(part_t_at_cell[:,8])
		cell_part_max_hi = max(part_t_at_cell[:,11])
		cell_part_max_Vi = max(part_t_at_cell[:,12])
		cell_part_max_Pi = max(part_t_at_cell[:,13])
		return (rr, cc, cell_part_max_eli, cell_part_max_ui, cell_part_max_hi, cell_part_max_Vi, cell_part_max_Pi)
	else:
		return None

def plot_SPEC_debris_map_v6_0(folder_path, flowpath_file_name, part_data, cluster_data, output_summary, road_xy_list, plot_naming, max_limits=[20, 10, 2_000, 1000, 2_000, 20, 10, 10, 1000], open_html=True, marker_size=5, line_width=2, layout_width=1000, layout_height=1000, max_cpu_num=8):
	'''
	2D map plot

	1. flowpath of debris-flow clusters
	2. debris-flow particles at each time-step
	3. debris-flow particle + cluster
	'''

	##############################################################
	## prefined features
	##############################################################

	if plot_naming == None:
		plot_naming = 'SPEC-debris'

	# black, green, blue, magenta, red, yellow, cyan
	color_list = ['rgba(0, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 0, 255, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 1)', 'rgba(0, 255, 255, 1)']
	color_list_ref = ['rgba(0, 0, 0, 0.75)', 'rgba(0, 255, 0, 0.75)', 'rgba(0, 0, 255, 0.75)', 'rgba(255, 0, 255, 0.75)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 0.75)', 'rgba(0, 255, 255, 0.75)']

	max_resolution = [2000, 5, 0.5, 50, 400]  # volume, velocity, depth, distance, pressure
	# plot_2D_max_limits - cluster u, h, V, D, P, particle u, h, V, P

	colorscale_data= [[0.0, 'rgba(0,0,0,1)'], [0.1, 'rgba(155,0,155,1)'], [0.2, 'rgba(255,0,255,1)'], [0.3, 'rgba(146,0,255,1)'], [0.4, 'rgba(0,0,255,1)'], [0.5, 'rgba(0,104,255,1)'], [0.6, 'rgba(0,255,220,1)'], [0.7, 'rgba(0,255,0,1)'], [0.8, 'rgba(255,255,0,1)'], [0.9, 'rgba(255,130,0,1)'], [1.0, 'rgba(255,0,0,1)']]

	colorscale_heat_data= [[0.0, 'rgba(255,255,255,0)'], [0.1, 'rgba(155,0,155,1)'], [0.2, 'rgba(255,0,255,1)'], [0.3, 'rgba(146,0,255,1)'], [0.4, 'rgba(0,0,255,1)'], [0.5, 'rgba(0,104,255,1)'], [0.6, 'rgba(0,255,220,1)'], [0.7, 'rgba(0,255,0,1)'], [0.8, 'rgba(255,255,0,1)'], [0.9, 'rgba(255,130,0,1)'], [1.0, 'rgba(255,0,0,1)']]

	# countour_scale = [[0.0, 'rgba(255,255,255,1)'], [0.1, 'rgba(255,255,255,1)'], [0.2, 'rgba(255,255,255,1)'], [0.3, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,1)'], [0.5, 'rgba(255,255,255,1)'], [0.6, 'rgba(255,255,255,1)'], [0.7, 'rgba(255,255,255,1)'], [0.8, 'rgba(255,255,255,1)'], [0.9, 'rgba(255,255,255,1)'], [1.0, 'rgba(255,255,255,1)']]

	countour_scale = [[0.0, 'rgba(255,255,255,0.0)'], [0.1, 'rgba(255,255,255,0.0)'], [0.2, 'rgba(255,255,255,0.0)'], [0.3, 'rgba(255,255,255,0.0)'], [0.4, 'rgba(255,255,255,0.0)'], [0.5, 'rgba(255,255,255,0.0)'], [0.6, 'rgba(255,255,255,0.0)'], [0.7, 'rgba(255,255,255,0.0)'], [0.8, 'rgba(255,255,255,0.0)'], [0.9, 'rgba(255,255,255,0.0)'], [1.0, 'rgba(255,255,255,0.0)']]
  
	##############################################################
	## 2D contour map
	##############################################################

	DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True)
		
	topoContour = go.Contour(
		x=gridUniqueX,
		y=gridUniqueY,
		z=DEM,
		line=dict(smoothing=0.85),
		autocontour=False, 
		colorscale=countour_scale,
		showscale=False,
		contours=dict(
			showlabels = True,
			labelfont = dict(
				family = 'Raleway',
				size = 12,
				color = 'black'
			)
			#cmax=np.ceil(max(flowpath.transpose()[loop])),
			#cmin=np.floor(min(flowpath.transpose()[loop]))
		)
	)

	plot_zmax = float(np.ceil(np.max(DEM) + max(max_limits[1], max_limits[6])))

	##############################################################
	## goal vector
	##############################################################
	if road_xy_list != None:
		goal_x = [road_xy_list[0][0], road_xy_list[1][0]]
		goal_y = [road_xy_list[0][1], road_xy_list[1][1]]
		goal_plot = go.Scatter(
			x=goal_x,
			y=goal_y,
			name='goal',
			mode='lines',
			line=dict(
				width=line_width,
				color='rgba(255, 165, 0, 1)'  # orange
			)
		)

	##############################################################
	## flowpath cluster map
	##############################################################
	if cluster_data != None:

		###############################
		## dataframe of all cluster data
		###############################
		cluster_df_list = []
		for cluster_num in range(len(cluster_data)):
			# 'cID,t,s,x,y,z,u,h,V,D,P,merged'
			# numpy array form for the current flowpath cluster
			cluster_array_i = np.array(cluster_data[int(cluster_num)])
			
			# dataframe for each particle
			cluster_df = pd.DataFrame(cluster_array_i, columns=['cID','t','s','x','y','z','u','h','V','D','P','A','Fr','CCHa','merged'])
			cluster_df_list.append(cluster_df)
		cluster_df = pd.concat(cluster_df_list)
		cluster_df_array = cluster_df.to_numpy()

		###############################
		## dataframe of all cluster data
		###############################
		
		if road_xy_list != None:
			data_path_flowpath = [topoContour, goal_plot]
			data_path_u = [topoContour, goal_plot]
			data_path_h = [topoContour, goal_plot]
			data_path_V = [topoContour, goal_plot]
			data_path_D = [topoContour, goal_plot]
			data_path_P = [topoContour, goal_plot]
		else:
			data_path_flowpath = [topoContour]
			data_path_u = [topoContour]
			data_path_h = [topoContour]
			data_path_V = [topoContour]
			data_path_D = [topoContour]
			data_path_P = [topoContour]

		###############################
		## cluster data 
		###############################
		for loop_cID in range(len(cluster_data)):

			###############################
			## SPEC-debris 
			###############################
			# 'cID,t,s,x,y,z,u,h,V,D,P,merged'
			cluster_array_i = np.array(cluster_data[loop_cID])
			
			# x,y-coordinates
			path_x = cluster_array_i[:,3] 
			path_y = cluster_array_i[:,4] 

			# elevation (z), depth(hi), volume (Vi), velocity (ui), distance from road (Di), dynamic pressure (Pi)
			path_time = cluster_array_i[:,1].tolist()
			path_s = cluster_array_i[:,2].tolist()
			path_Z = cluster_array_i[:,5].tolist()
			path_ui = cluster_array_i[:,6].tolist()
			path_hi = cluster_array_i[:,7].tolist()
			path_Vi = cluster_array_i[:,8].tolist()
			path_Di = cluster_array_i[:,9].tolist()
			path_Pi = cluster_array_i[:,10].tolist()

			###############################
			## data_path_flowpath 
			###############################
			hovertext_cluster_flowpath = [(round(tt,2),round(ss,2),round(zz,2),round(uu,2),round(hh,2),round(VV,2),round(DD,2),round(PP,2)) for tt,ss,zz,uu,hh,VV,DD,PP in zip(path_time, path_s, path_Z, path_ui, path_hi, path_Vi, path_Di, path_Pi)]
			cluster_plot_xy = go.Scatter(
				x=path_x,
				y=path_y,
				name='SPEC-debris-flowpath-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_flowpath,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=color_list[loop_cID]
				)
			)
			data_path_flowpath.append(cluster_plot_xy)

			###############################
			## data_path_u 
			###############################
			hovertext_cluster_u = [(round(tt,2),round(zz,2),round(uu,2)) for tt,zz,uu in zip(path_time, path_Z, path_ui)]
			cluster_plot_u = go.Scatter(
				x=path_x,
				y=path_y,
				name='SPEC-debris-speed-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_u,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,6],decimals=2),
					colorscale=colorscale_data,
					colorbar=dict(
						title='velocity [m/s]',
						ticks="outside",
					),
					cmax=max_limits[0],
					cmin=0
				)
			)
			data_path_u.append(cluster_plot_u)

			###############################
			## data_path_h
			###############################
			hovertext_cluster_h = [(round(tt,2),round(zz,2),round(hh,2)) for tt,zz,hh in zip(path_time, path_Z, path_hi)]
			cluster_plot_h = go.Scatter(
				x=path_x,
				y=path_y,
				name='SPEC-debris-depth-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_h,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,7],decimals=2),
					colorscale=colorscale_data,
					colorbar=dict(
						title='depth [m]',
						ticks="outside",
					),
					cmax=max_limits[1],
					cmin=0
				)
			)
			data_path_h.append(cluster_plot_h)

			###############################
			## data_path_V 
			###############################
			hovertext_cluster_V = [(round(tt,2),round(zz,2),round(VV,3)) for tt,zz,VV in zip(path_time, path_Z, path_Vi)]
			cluster_plot_V = go.Scatter(
				x=path_x,
				y=path_y,
				name='SPEC-debris-volume-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_V,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,8],decimals=2),
					colorscale=colorscale_data,
					colorbar=dict(
						title='volume [m^3]',
						ticks="outside",
					),
					cmax=max_limits[2],
					cmin=0
				)
			)
			data_path_V.append(cluster_plot_V)

			###############################
			## data_path_D
			###############################
			hovertext_cluster_D = [(round(tt,2),round(zz,2),round(DD,2)) for tt,zz,DD in zip(path_time, path_Z, path_Di)]
			cluster_plot_D = go.Scatter(
				x=path_x,
				y=path_y,
				name='SPEC-debris-dist-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_D,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,9],decimals=2),
					colorscale=colorscale_data,
					colorbar=dict(
						title='dist. [m]',
						ticks="outside",
					),
					cmax=max_limits[3],
					cmin=0
				)
			)
			data_path_D.append(cluster_plot_D)

			###############################
			## data_path_P 
			###############################
			hovertext_cluster_P = [(round(tt,2),round(zz,2),round(PP,2)) for tt,zz,PP in zip(path_time, path_Z, path_Pi)]
			cluster_plot_P = go.Scatter(
				x=path_x,
				y=path_y,
				name='SPEC-debris-pressure-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_P,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,10],decimals=2),
					colorscale=colorscale_data,
					colorbar=dict(
						title='press. [kPa]',
						ticks="outside",
					),
					cmax=max_limits[4],
					cmin=0
				)
			)
			data_path_P.append(cluster_plot_P)

		###############################
		## overall data summary
		###############################
		
		output_array_i = np.array(output_summary)
		
		# SPEC-debris - output - 't,s,x,y,z,u,h,V'
		# x,y-coordinates
		overall_x = output_array_i[:,2] 
		overall_y = output_array_i[:,3] 

		# elevation (z), volume (Vi), velocity (ui), depth (hi)
		overall_time = output_array_i[:,0].tolist()
		overall_s = output_array_i[:,1].tolist()
		overall_Z = output_array_i[:,4].tolist()
		overall_ui = output_array_i[:,5].tolist()
		overall_hi = output_array_i[:,6].tolist()
		overall_Vi = output_array_i[:,7].tolist()

		hovertext_flowpath_overall = [(round(tt,2),round(ss,2),round(zz,2),round(uu,2),round(hh,2),round(VV,2)) for tt,ss,zz,uu,hh,VV in zip(overall_time, overall_s, overall_Z, overall_ui, overall_hi, overall_Vi)]
		output_plot_xy = go.Scatter(
			x=overall_x,
			y=overall_y,
			name='SPEC-debris-overall', 
			hovertext=hovertext_flowpath_overall,
			mode='lines',
			line=dict(
				width=line_width,
				color=color_list[0] 
			)
		)
		data_path_flowpath.append(output_plot_xy)

		###############################
		## layout
		###############################
		# data_path_flowpath
		layout_map_flowpath = go.Layout(
			title=plot_naming + ' - 2D map - flowpath',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_velocity
		layout_map_u = go.Layout(
			title=plot_naming + ' - 2D map - velocity',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_depth
		layout_map_h = go.Layout(
			title=plot_naming + ' - 2D map - depth',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_volume
		layout_map_V = go.Layout(
			title=plot_naming + ' - 2D map - volume',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_distane-from-road
		layout_map_D = go.Layout(
			title=plot_naming + ' - 2D map - distance-from-road',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_pressure
		layout_map_P = go.Layout(
			title=plot_naming + ' - 2D map - impact pressure',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		###############################
		## plot 
		###############################

		# figure
		fig_map_flowpath = go.Figure(data=data_path_flowpath, layout=layout_map_flowpath)
		fig_map_u = go.Figure(data=data_path_u, layout=layout_map_u)
		fig_map_h = go.Figure(data=data_path_h, layout=layout_map_h)
		fig_map_V = go.Figure(data=data_path_V, layout=layout_map_V)
		fig_map_D = go.Figure(data=data_path_D, layout=layout_map_D)
		fig_map_P = go.Figure(data=data_path_P, layout=layout_map_P)

		# plot into html
		plot(fig_map_flowpath, filename=folder_path+plot_naming+' - 2D map - flowpath.html', auto_open=open_html)
		plot(fig_map_u, filename=folder_path+plot_naming+' - 2D map - velocity.html', auto_open=open_html)
		plot(fig_map_h, filename=folder_path+plot_naming+' - 2D map - depth.html', auto_open=open_html)
		plot(fig_map_V, filename=folder_path+plot_naming+' - 2D map - volume.html', auto_open=open_html)
		plot(fig_map_D, filename=folder_path+plot_naming+' - 2D map - distance_from_road.html', auto_open=open_html)
		plot(fig_map_P, filename=folder_path+plot_naming+' - 2D map - impact pressure.html', auto_open=open_html)

	##############################################################
	## flowpath particle map - particles at each time
	##############################################################
	if part_data != None:

		###############################
		## particle data - sorted for each timestep
		###############################

		# dataframe of all particles data
		part_df_list = []
		pID_list = np.arange(len(part_data[0])).tolist()
		pID_col_array = np.transpose(np.array([pID_list]))

		for part_t in range(len(part_data)):

			# name file names
			# 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P' -> 'pID,cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P,i,j'

			# numpy array form for the current flowpath cluster
			parts_data_i = np.array(part_data[part_t])

			# row (i) and column (j) that the particle is occupying
			part_row_col = np.array([compute_ij(part_d[3], part_d[4], gridUniqueX, gridUniqueY, deltaX, deltaY) for part_d in part_data[part_t]])

			# stack part_num to the part_array_i
			part_array_ii = np.hstack((pID_col_array, parts_data_i, part_row_col))

			# dataframe for each particle
			part_df = pd.DataFrame(part_array_ii, columns=['pID','cID','t','s','x','y','z','elevation','u','ux','uy','h','V','P','i','j'])

			part_df_list.append(part_df)

		part_df = pd.concat(part_df_list)

		# total simulation time-step
		part_t_all_list = part_df['t'].values.tolist()
		part_t_list = np.unique(part_t_all_list).tolist()
		part_cID_list = np.unique(part_df['cID'].values.tolist()).tolist()
		part_df_array = part_df.to_numpy()

		# road
		if road_xy_list != None:
			data_part_el = [topoContour, goal_plot]
			data_part_u = [topoContour, goal_plot]
			data_part_h = [topoContour, goal_plot]
			data_part_V = [topoContour, goal_plot]
			data_part_P = [topoContour, goal_plot]
		else:
			data_part_el = [topoContour]
			data_part_u = [topoContour]
			data_part_h = [topoContour]
			data_part_V = [topoContour]
			data_part_P = [topoContour]

		###############################
		## max value at each DEM cell
		###############################
		# 'pID0','cID1','t2','s3','x4','y5','z6','elevation7','u8','ux9','uy10','h11','V12','P13','i14','j15'
		
		# go through each line of csv file and place them into a grid format
		# row number = y-coordinates
		# col number = x-coordinates
		part_max_el_mesh = np.zeros((len(gridUniqueY),len(gridUniqueX)),dtype=float)
		part_max_ui_mesh = np.zeros((len(gridUniqueY),len(gridUniqueX)),dtype=float)
		part_max_hi_mesh = np.zeros((len(gridUniqueY),len(gridUniqueX)),dtype=float)
		part_max_Vi_mesh = np.zeros((len(gridUniqueY),len(gridUniqueX)),dtype=float)
		part_max_Pi_mesh = np.zeros((len(gridUniqueY),len(gridUniqueX)),dtype=float)

		part_data_cell_input = []
		for rr in range(len(gridUniqueY)):
			for cc in range(len(gridUniqueX)):
				part_data_cell_input.append((part_df, rr, cc))

		## multiprocessing set-up
		if mp.cpu_count() >= max_cpu_num:
			cpu_num = max_cpu_num 
		else:
			cpu_num = mp.cpu_count()
		pool_part = mp.Pool(cpu_num)
		
		# max cell value from particle data
		max_cell_value = pool_part.map(max_part_data_at_cell_MP, part_data_cell_input)

		## stop multiprocessing
		pool_part.close()
		pool_part.join()

		for max_cell_value_i in max_cell_value:
			if max_cell_value_i is not None:
				rr, cc, cell_part_max_eli, cell_part_max_ui, cell_part_max_hi, cell_part_max_Vi, cell_part_max_Pi = max_cell_value_i
				part_max_el_mesh[rr][cc] = cell_part_max_eli
				part_max_ui_mesh[rr][cc] = cell_part_max_ui
				part_max_hi_mesh[rr][cc] = cell_part_max_hi
				part_max_Vi_mesh[rr][cc] = cell_part_max_Vi
				part_max_Pi_mesh[rr][cc] = cell_part_max_Pi

		###############################
		## data_part_elevation
		###############################
		plot_part_max_elevation = go.Heatmap(
			x=gridUniqueX,
			y=gridUniqueY,
			z=part_max_el_mesh, 
			name='SPEC-debris-particle-max-elevation', 
			hovertemplate=
				"x: %{x:.2f}<br>" +
				"y: %{y:.2f}<br>" +
				"elevation: %{z:.2f}" +
				"<extra></extra>",
			colorbar=dict(
				title='elevation [m]',
				ticks="outside"
			),
			zmax=plot_zmax, 
			zmin=0,
			zsmooth=False,	#  "fast" | "best" | False  https://chart-studio.plotly.com/~plotly.js/637.embed
			colorscale=colorscale_heat_data
		)
		data_part_el.append(plot_part_max_elevation)

		###############################
		## data_part_u
		###############################
		plot_part_max_u = go.Heatmap(
			x=gridUniqueX,
			y=gridUniqueY,
			z=part_max_ui_mesh, 
			name='SPEC-debris-particle-max-velocity', 
			hovertemplate=
				"x: %{x:.2f}<br>" +
				"y: %{y:.2f}<br>" +
				"u: %{z:.2f}" +
				"<extra></extra>",
			colorbar=dict(
				title='velocity [m/s]',
				ticks="outside"
			),
			zmax=max_limits[5], 
			zmin=0,
			zsmooth=False,	#  "fast" | "best" | False  https://chart-studio.plotly.com/~plotly.js/637.embed
			colorscale=colorscale_heat_data
		)
		data_part_u.append(plot_part_max_u)

		###############################
		## data_part_h
		###############################
		plot_part_max_h = go.Heatmap(
			x=gridUniqueX,
			y=gridUniqueY,
			z=part_max_hi_mesh, 
			name='SPEC-debris-particle-max-depth', 
			hovertemplate=
				"x: %{x:.2f}<br>" +
				"y: %{y:.2f}<br>" +
				"h: %{z:.2f}" +
				"<extra></extra>",
			colorbar=dict(
				title='depth [m]',
				ticks="outside"
			),
			zmax=max_limits[6], 
			zmin=0,
			zsmooth=False,		#  "fast" | "best" | False  https://chart-studio.plotly.com/~plotly.js/637.embed
			colorscale=colorscale_heat_data
		)
		data_part_h.append(plot_part_max_h)

		###############################
		## data_part_V
		###############################
		plot_part_max_V = go.Heatmap(
			x=gridUniqueX,
			y=gridUniqueY,
			z=part_max_Vi_mesh, 
			name='SPEC-debris-particle-max-volume',
			hovertemplate=
				"x: %{x:.2f}<br>" +
				"y: %{y:.2f}<br>" +
				"V: %{z:.2f}" +
				"<extra></extra>",
			colorbar=dict(
				title='volume [m^3]',
				ticks="outside"
			),
			zmax=max_limits[7], 
			zmin=0, 
			zsmooth=False,		#  "fast" | "best" | False  https://chart-studio.plotly.com/~plotly.js/637.embed
			colorscale=colorscale_heat_data
		)
		data_part_V.append(plot_part_max_V)

		###############################
		## data_part_P
		###############################
		plot_part_max_P = go.Heatmap(
			x=gridUniqueX,
			y=gridUniqueY,
			z=part_max_Pi_mesh, 
			name='SPEC-debris-particle-max-pressure', 
			hovertemplate=
				"x: %{x:.2f}<br>" +
				"y: %{y:.2f}<br>" +
				"P: %{z:.2f}" +
				"<extra></extra>",
			colorbar=dict(
				title='impact pressure [kPa]',
				ticks="outside"
			),
			zmax=max_limits[8], 
			zmin=0,
			zsmooth=False,		#  "fast" | "best" | False  https://chart-studio.plotly.com/~plotly.js/637.embed
			colorscale=colorscale_heat_data
		)
		data_part_P.append(plot_part_max_P)

		###############################
		## layout
		###############################
		# part_el
		layout_part_el = go.Layout(
			title=plot_naming + ' - 2D particle - elevation',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)

		# part_u
		layout_part_u = go.Layout(
			title=plot_naming + ' - 2D particle - velocity',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)
		
		# part_h
		layout_part_h = go.Layout(
			title=plot_naming + ' - 2D particle - depth',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)
		
		# part_V 
		layout_part_V = go.Layout(
			title=plot_naming + ' - 2D particle - volume',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)

		layout_part_P = go.Layout(
			title=plot_naming + ' - 2D particle - pressure',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)
		
		###############################
		## plot 
		###############################

		# figure
		fig_part_el = go.Figure(data=data_part_el, layout=layout_part_el)
		fig_part_u = go.Figure(data=data_part_u, layout=layout_part_u)
		fig_part_h = go.Figure(data=data_part_h, layout=layout_part_h)
		fig_part_V = go.Figure(data=data_part_V, layout=layout_part_V)
		fig_part_P = go.Figure(data=data_part_P, layout=layout_part_P)

		# plot into html
		plot(fig_part_el, filename=folder_path+plot_naming+' - 2D max elevation.html', auto_open=open_html)
		plot(fig_part_u, filename=folder_path+plot_naming+' - 2D max velocity.html', auto_open=open_html)
		plot(fig_part_h, filename=folder_path+plot_naming+' - 2D max depth.html', auto_open=open_html)
		plot(fig_part_V, filename=folder_path+plot_naming+' - 2D max volume.html', auto_open=open_html)
		plot(fig_part_P, filename=folder_path+plot_naming+' - 2D max pressure.html', auto_open=open_html)

	##############################################################
	## flowpath particle + cluster map
	##############################################################
	if cluster_data != None and part_data != None:

		###############################
		## add data from previous plotly cluster and particle plots
		###############################
		# road
		if road_xy_list != None:
			# background - contour and road
			data_part_cluster_el = [topoContour, goal_plot]
			data_part_cluster_u = [topoContour, goal_plot]
			data_part_cluster_h = [topoContour, goal_plot]
			data_part_cluster_V = [topoContour, goal_plot]
			data_part_cluster_P = [topoContour, goal_plot]

			# add cluster data
			for cID_Data in data_path_flowpath[2:]:
				data_part_cluster_el.append(cID_Data)
				data_part_cluster_u.append(cID_Data)
				data_part_cluster_h.append(cID_Data)
				data_part_cluster_V.append(cID_Data)
				data_part_cluster_P.append(cID_Data)

		else:
			# background - contour
			data_part_cluster_el = [topoContour]
			data_part_cluster_u = [topoContour]
			data_part_cluster_h = [topoContour]
			data_part_cluster_V = [topoContour]
			data_part_cluster_P = [topoContour]

			# add cluster data
			for cID_Data in data_path_flowpath[1:]:
				data_part_cluster_el.append(cID_Data)
				data_part_cluster_u.append(cID_Data)
				data_part_cluster_h.append(cID_Data)
				data_part_cluster_V.append(cID_Data)
				data_part_cluster_P.append(cID_Data)

		# add particle heat map 
		for p_el, p_u, p_h, p_V, p_P in zip(data_part_el, data_part_u, data_part_h, data_part_V, data_part_P):
			data_part_cluster_el.append(p_el)
			data_part_cluster_u.append(p_u)
			data_part_cluster_h.append(p_h)
			data_part_cluster_V.append(p_V)
			data_part_cluster_P.append(p_P)
		
		###############################
		## layout
		###############################
		# velocity
		layout_part_cluster_el = go.Layout(
			title=plot_naming + ' - 2D particle cluster - elevation',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)

		# velocity
		layout_part_cluster_u = go.Layout(
			title=plot_naming + ' - 2D particle cluster - velocity',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)

		# depth
		layout_part_cluster_h = go.Layout(
			title=plot_naming + ' - 2D particle cluster - depth',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)

		# volume
		layout_part_cluster_V = go.Layout(
			title=plot_naming + ' - 2D particle cluster - volume',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)

		# impact pressure
		layout_part_cluster_P = go.Layout(
			title=plot_naming + ' - 2D particle cluster - pressure',
			paper_bgcolor='rgba(255,255,255,1)',
			plot_bgcolor='rgba(255,255,255,1)',
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)
		
		###############################
		## plot 
		###############################

		# figure
		fig_part_cluster_el = go.Figure(data=data_part_cluster_el, layout=layout_part_cluster_el)
		fig_part_cluster_u = go.Figure(data=data_part_cluster_u, layout=layout_part_cluster_u)
		fig_part_cluster_h = go.Figure(data=data_part_cluster_h, layout=layout_part_cluster_h)
		fig_part_cluster_V = go.Figure(data=data_part_cluster_V, layout=layout_part_cluster_V)
		fig_part_cluster_P = go.Figure(data=data_part_cluster_P, layout=layout_part_cluster_P)

		# plot into html
		plot(fig_part_cluster_el, filename=folder_path+plot_naming+' - 2D particle cluster - elevation.html', auto_open=open_html)
		plot(fig_part_cluster_u, filename=folder_path+plot_naming+' - 2D particle cluster - velocity.html', auto_open=open_html)
		plot(fig_part_cluster_h, filename=folder_path+plot_naming+' - 2D particle cluster - depth.html', auto_open=open_html)
		plot(fig_part_cluster_V, filename=folder_path+plot_naming+' - 2D particle cluster - volume.html', auto_open=open_html)
		plot(fig_part_cluster_P, filename=folder_path+plot_naming+' - 2D particle cluster - pressure.html', auto_open=open_html)

	return None

def plot_SPEC_debris_surface_v4_0(folder_path, flowpath_file_name, part_data, cluster_data, plot_naming, max_limits=[20, 10, 2_000, 1000, 2_000, 20, 10, 10, 1000], open_html=True, z_offset=0, marker_size=5, line_width=2, layout_width=1000, layout_height=1000):
	'''
	3D surface plot
	'''
	##############################################################
	## prefined features
	##############################################################

	if plot_naming == None:
		plot_naming = 'SPEC-debris'

	# black, green, blue, magenta, red, yellow, cyan
	color_list = ['rgba(0, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 0, 255, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 1)', 'rgba(0, 255, 255, 1)']
	color_list_ref = ['rgba(0, 0, 0, 0.75)', 'rgba(0, 255, 0, 0.75)', 'rgba(0, 0, 255, 0.75)', 'rgba(255, 0, 255, 0.75)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 0.75)', 'rgba(0, 255, 255, 0.75)']

	max_resolution = [2000, 5, 0.5, 50, 400]  # volume, velocity, depth, distance, pressure
	
	colorscale_data= [[0.0, 'rgba(0,0,0,1)'], [0.1, 'rgba(155,0,155,1)'], [0.2, 'rgba(255,0,255,1)'], [0.3, 'rgba(146,0,255,1)'], [0.4, 'rgba(0,0,255,1)'], [0.5, 'rgba(0,104,255,1)'], [0.6, 'rgba(0,255,220,1)'], [0.7, 'rgba(0,255,0,1)'], [0.8, 'rgba(255,255,0,1)'], [0.9, 'rgba(255,130,0,1)'], [1.0, 'rgba(255,0,0,1)']]

	countour_scale = [[0.0, 'rgba(255,255,255,1)'], [0.1, 'rgba(255,255,255,1)'], [0.2, 'rgba(255,255,255,1)'], [0.3, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,1)'], [0.5, 'rgba(255,255,255,1)'], [0.6, 'rgba(255,255,255,1)'], [0.7, 'rgba(255,255,255,1)'], [0.8, 'rgba(255,255,255,1)'], [0.9, 'rgba(255,255,255,1)'], [1.0, 'rgba(255,255,255,1)']]
  
	##############################################################
	## 3D surface map
	##############################################################

	DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True)
		
	topoSurface = go.Surface(
		x=gridUniqueX,
		y=gridUniqueY,
		z=DEM,
		colorscale = 'geyser',
		showscale = False,
		# contours = {
		# 	"x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color":"black"},
		# 	"z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
		# }
	)

	plot_zmax = float(np.ceil(np.max(DEM) + max(max_limits[1], max_limits[6])))

	##############################################################
	## flowpath cluster map
	##############################################################
	if cluster_data != None:

		###############################
		## dataframe of all cluster data
		###############################
		cluster_df_list = []
		for cluster_num in range(len(cluster_data)):
			# 'cID,t,s,x,y,z,u,h,V,D,P,merged'
			# numpy array form for the current flowpath cluster
			cluster_array_i = np.array(cluster_data[int(cluster_num)])
			
			# dataframe for each particle
			cluster_df = pd.DataFrame(cluster_array_i, columns=['cID','t','s','x','y','z','u','h','V','D','P','A','Fr','CCHa','merged'])
			cluster_df_list.append(cluster_df)
		cluster_df = pd.concat(cluster_df_list)

		###############################
		## cluster data 
		###############################
		
		data_path_flowpath_3D = [topoSurface]
		data_path_u_3D = [topoSurface]
		data_path_h_3D = [topoSurface]
		data_path_V_3D = [topoSurface]
		data_path_D_3D = [topoSurface]
		data_path_P_3D = [topoSurface]

		for loop_cID in range(len(cluster_data)):

			###############################
			## SPEC-debris 
			###############################
			# 'cID,t,s,x,y,z,u,h,V,D,P,merged'
			cluster_array_i = np.array(cluster_data[loop_cID])
			
			# x,y,z-coordinates
			path_x = cluster_array_i[:,3] 
			path_y = cluster_array_i[:,4] 
			path_z_plot = cluster_array_i[:,5] + z_offset

			# volume (Vi), velocity (ui), distance from road (Di), dynamic pressure (Pi)
			path_time = cluster_array_i[:,1].tolist()
			path_z = cluster_array_i[:,5].tolist()
			path_s = cluster_array_i[:,2].tolist()
			path_ui = cluster_array_i[:,6].tolist()
			path_hi = cluster_array_i[:,7].tolist()
			path_Vi = cluster_array_i[:,8].tolist()
			path_Di = cluster_array_i[:,9].tolist()
			path_Pi = cluster_array_i[:,10].tolist()

			###############################
			## data_path_flowpath 
			###############################
			hovertext_cluster_flowpath_3D = [(round(tt,2),round(zz,2),round(ss,2),round(uu,2),round(hh,2),round(VV,2),round(DD,2),round(PP,2)) for tt,zz,ss,uu,hh,VV,DD,PP in zip(path_time, path_z, path_s, path_ui, path_hi, path_Vi, path_Di, path_Pi)]
			cluster_plot_xy = go.Scatter3d(
				x=path_x,
				y=path_y,
				z=path_z_plot,
				name='SPEC-debris-flowpath-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_flowpath_3D,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=color_list[loop_cID]
				)
			)
			data_path_flowpath_3D.append(cluster_plot_xy)

			###############################
			## data_path_u 
			###############################
			hovertext_cluster_u_3D = [(round(tt,2),round(zz,2),round(uu,2)) for tt,zz,uu in zip(path_time, path_z, path_ui)]
			cluster_plot_u = go.Scatter3d(
				x=path_x,
				y=path_y,
				z=path_z_plot,
				name='SPEC-debris-velocity-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_u_3D,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,6],decimals=2),
					colorscale='Viridis',
					colorbar=dict(
						title='velocity [m/s]',
						ticks="outside",
					),
					cmax=max_limits[0],
					cmin=0
				)
			)
			data_path_u_3D.append(cluster_plot_u)

			###############################
			## data_path_h
			###############################
			hovertext_cluster_h_3D = [(round(tt,2),round(zz,2),round(hh,2)) for tt,zz,hh in zip(path_time, path_z, path_hi)]
			cluster_plot_h = go.Scatter3d(
				x=path_x,
				y=path_y,
				z=path_z_plot,
				name='SPEC-debris-depth-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_h_3D,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,7],decimals=2),
					colorscale='Viridis',
					colorbar=dict(
						title='depth [m]',
						ticks="outside",
					),
					cmax=max_limits[1],
					cmin=0
				)
			)
			data_path_h_3D.append(cluster_plot_h)

			###############################
			## data_path_V 
			###############################
			hovertext_cluster_V_3D = [(round(tt,2),round(zz,2),round(VV,3)) for tt,zz,VV in zip(path_time, path_z, path_Vi)]
			cluster_plot_V = go.Scatter3d(
				x=path_x,
				y=path_y,
				z=path_z_plot,
				name='SPEC-debris-volume-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_V_3D,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,8],decimals=2),
					colorscale='Viridis',
					colorbar=dict(
						title='volume [m^3]',
						ticks="outside",
					),
					cmax=max_limits[2],
					cmin=0
				)
			)
			data_path_V_3D.append(cluster_plot_V)

			###############################
			## data_path_D
			###############################
			hovertext_cluster_D_3D = [(round(tt,2),round(zz,2),round(DD,2)) for tt,zz,DD in zip(path_time, path_z, path_Di)]
			cluster_plot_D = go.Scatter3d(
				x=path_x,
				y=path_y,
				z=path_z_plot,
				name='SPEC-debris-dist-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_D_3D,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,9],decimals=2),
					colorscale='Viridis',
					colorbar=dict(
						title='dist. [m]',
						ticks="outside",
					),
					cmax=max_limits[3],
					cmin=0
				)
			)
			data_path_D_3D.append(cluster_plot_D)

			###############################
			## data_path_P 
			###############################
			hovertext_cluster_P_3D = [(round(tt,2),round(zz,2),round(PP,2)) for tt,zz,PP in zip(path_time, path_z, path_Pi)]
			cluster_plot_P = go.Scatter3d(
				x=path_x,
				y=path_y,
				z=path_z_plot,
				name='SPEC-debris-pressure-cluster'+str(loop_cID), 
				hovertext=hovertext_cluster_P_3D,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size,
					color=np.round(cluster_array_i[:,10],decimals=2),
					colorscale='Viridis',
					colorbar=dict(
						title='press. [kPa]',
						ticks="outside",
					),
					cmax=max_limits[4],
					cmin=0
				)
			)
			data_path_P_3D.append(cluster_plot_P)

		###############################
		## layout
		###############################
		# data_path_flowpath
		layout_map_flowpath_3D = go.Layout(
			title=plot_naming + ' - 3D surface - flowpath',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_velocity
		layout_map_u_3D = go.Layout(
			title=plot_naming + ' - 3D surface - velocity',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_depth
		layout_map_h_3D = go.Layout(
			title=plot_naming + ' - 3D surface - depth',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_volume
		layout_map_V_3D = go.Layout(
			title=plot_naming + ' - 3D surface - volume',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_distance-from-road
		layout_map_D_3D = go.Layout(
			title=plot_naming + ' - 3D surface - distance_from_road',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# data_path_pressure
		layout_map_P_3D = go.Layout(
			title=plot_naming + ' - 3D surface - impact pressure',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		###############################
		## plot 
		###############################

		# figure
		fig_map_flowpath_3D = go.Figure(data=data_path_flowpath_3D, layout=layout_map_flowpath_3D)
		fig_map_u_3D = go.Figure(data=data_path_u_3D, layout=layout_map_u_3D)
		fig_map_h_3D = go.Figure(data=data_path_h_3D, layout=layout_map_h_3D)
		fig_map_V_3D = go.Figure(data=data_path_V_3D, layout=layout_map_V_3D)
		fig_map_D_3D = go.Figure(data=data_path_D_3D, layout=layout_map_D_3D)
		fig_map_P_3D = go.Figure(data=data_path_P_3D, layout=layout_map_P_3D)

		# plot into html
		plot(fig_map_flowpath_3D, filename=folder_path+plot_naming+' - 3D surface - flowpath.html', auto_open=open_html)
		plot(fig_map_u_3D, filename=folder_path+plot_naming+' - 3D surface - velocity.html', auto_open=open_html)
		plot(fig_map_h_3D, filename=folder_path+plot_naming+' - 3D surface - depth.html', auto_open=open_html)
		plot(fig_map_V_3D, filename=folder_path+plot_naming+' - 3D surface - volume.html', auto_open=open_html)
		plot(fig_map_D_3D, filename=folder_path+plot_naming+' - 3D surface - distance_from_road.html', auto_open=open_html)
		plot(fig_map_P_3D, filename=folder_path+plot_naming+' - 3D surface - impact pressure.html', auto_open=open_html)

		del data_path_flowpath_3D
		del data_path_u_3D
		del data_path_h_3D
		del data_path_V_3D
		del data_path_D_3D
		del data_path_P_3D

	##############################################################
	## flowpath particle map - particles at each time
	##############################################################
	if part_data != None:

		###############################
		## particle data - sorted for each timestep
		###############################

		# dataframe of all particles data
		part_df_list = []
		pID_list = np.arange(len(part_data[0])).tolist()
		pID_col_array = np.transpose(np.array([pID_list]))

		for part_t in range(len(part_data)):

			# 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P' -> 'pID,cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P

			# numpy array form for the current flowpath cluster
			part_array_ti = np.array(part_data[part_t])
			
			# stack part_num to the part_array_i
			part_array_ii = np.hstack((pID_col_array, part_array_ti))

			# dataframe for each particle
			part_df = pd.DataFrame(part_array_ii, columns=['pID','cID','t','s','x','y','z','elevation','u','ux','uy','h','V','P'])
			part_df_list.append(part_df)
		part_df = pd.concat(part_df_list)

		# total simulation time-step
		part_t_all_list = part_df['t'].values.tolist()
		part_t_list = np.unique(part_t_all_list).tolist()
		part_cID_list = np.unique(part_df['cID'].values.tolist()).tolist()
		part_df_array = part_df.to_numpy()

		###############################
		## SPEC-debris
		###############################
		# 'pID0','cID1','t2','s3','x4','y5','z6','elevation7','u8','ux9','uy10','h11','V12','P13','i14','j15'

		data_part_path_3D = [topoSurface]
		data_part_u_3D = [topoSurface]
		data_part_h_3D = [topoSurface]
		data_part_V_3D = [topoSurface]
		data_part_P_3D = [topoSurface]
		
		# x,y-coordinates
		part_x = part_df_array[:,4] 
		part_y = part_df_array[:,5] 
		part_z_plot = part_df_array[:,6] + z_offset 
		# part_z_plot = part_df_array[:,7]

		# other data
		part_pID = part_df_array[:,0].tolist()
		part_cID = part_df_array[:,1].tolist()
		part_t = part_df_array[:,2].tolist()
		part_s = part_df_array[:,3].tolist()
		part_Z = part_df_array[:,6].tolist()
		part_el = part_df_array[:,7].tolist()
		part_ui = part_df_array[:,8].tolist()
		part_uxi = part_df_array[:,9].tolist()
		part_uyi = part_df_array[:,10].tolist()
		part_hi = part_df_array[:,11].tolist()
		part_Vi = part_df_array[:,12].tolist() 			
		part_Pi = part_df_array[:,13].tolist() 			

		###############################
		## data_part_path
		###############################
		hovertext_part_path_3D = [(int(pp),int(cc),round(tt,2),round(ss,2),round(zz,2),round(ell,2),round(uu,3),round(uxx,3),round(uyy,3),round(hh,2),round(VV,2),round(PP,2)) for pp,cc,tt,ss,zz,ell,uu,uxx,uyy,hh,VV,PP in zip(part_pID, part_cID, part_t, part_s, part_Z, part_el, part_ui, part_uxi, part_uyi, part_hi, part_Vi, part_Pi)]
		plot_part_xy = go.Scatter3d(
			x=part_x,
			y=part_y,
			z=part_z_plot,
			name='SPEC-debris-particles', 
			hovertext=hovertext_part_path_3D,
			mode='markers',
			marker=dict(
				size=marker_size,
				color='rgba(0, 0, 0, 0.8)'
			)
		)
		data_part_path_3D.append(plot_part_xy)

		###############################
		## data_part_u
		###############################
		hovertext_part_u_3D = [(int(pp),int(cc),round(zz,2),round(ell,2),round(uu,3)) for pp,cc,zz,ell,uu in zip(part_pID, part_cID, part_Z, part_el, part_ui)]
		plot_part_u = go.Scatter3d(
			x=part_x,
			y=part_y,
			z=part_z_plot,
			name='SPEC-debris-velocity',  
			hovertext=hovertext_part_u_3D,
			mode='markers',
			marker=dict(
				size=marker_size,
				color=np.round(part_ui,decimals=2),
				colorscale=colorscale_data,
				colorbar=dict(
					title='velocity [m/s]',
					ticks="outside",
				),
				cmax=max_limits[5],
				cmin=0
			)
		)
		data_part_u_3D.append(plot_part_u)

		###############################
		## data_part_h
		###############################
		hovertext_part_h_3D = [(int(pp),int(cc),round(zz,2),round(ell,2),round(hh,3)) for pp,cc,zz,ell,hh in zip(part_pID, part_cID, part_Z, part_el, part_hi)]
		plot_part_h = go.Scatter3d(
			x=part_x,
			y=part_y,
			z=part_z_plot,
			name='SPEC-debris-depth',  
			hovertext=hovertext_part_u_3D,
			mode='markers',
			marker=dict(
				size=marker_size,
				color=np.round(part_ui,decimals=2),
				colorscale=colorscale_data,
				colorbar=dict(
					title='depth [m]',
					ticks="outside",
				),
				cmax=max_limits[6],
				cmin=0
			)
		)
		data_part_h_3D.append(plot_part_h)


		###############################
		## data_part_V
		###############################
		hovertext_part_V_3D = [(int(pp),int(cc),round(zz,2),round(ell,2),round(VV,3)) for pp,cc,zz,ell,VV in zip(part_pID, part_cID, part_Z, part_el, part_Vi)]
		plot_part_V = go.Scatter3d(
			x=part_x,
			y=part_y,
			z=part_z_plot,
			name='SPEC-debris-volume', 
			hovertext=hovertext_part_V_3D,
			mode='markers',
			marker=dict(
				size=marker_size,
				color=np.round(part_Vi,decimals=2),
				colorscale=colorscale_data,
				colorbar=dict(
					title='volume [m^3]',
					ticks="outside",
				),
				cmax=max_limits[7],
				cmin=0
			)
		)
		data_part_V_3D.append(plot_part_V)

		###############################
		## data_part_P
		###############################
		hovertext_part_P_3D = [(int(pp),int(cc),round(zz,2),round(ell,2),round(PP,3)) for pp,cc,zz,ell,PP in zip(part_pID, part_cID, part_Z, part_el, part_Pi)]
		plot_part_P = go.Scatter3d(
			x=part_x,
			y=part_y,
			z=part_z_plot,
			name='SPEC-debris-pressure', 
			hovertext=hovertext_part_P_3D,
			mode='markers',
			marker=dict(
				size=marker_size,
				color=np.round(part_Pi,decimals=2),
				colorscale=colorscale_data,
				colorbar=dict(
					title='pressure [kPa]',
					ticks="outside",
				),
				cmax=max_limits[8],
				cmin=0
			)
		)
		data_part_P_3D.append(plot_part_P)

		###############################
		## layout
		###############################
		# part_path
		layout_part_path_3D = go.Layout(
			title=plot_naming + ' - 3D particle - paths',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)
		
		# part_u
		layout_part_u_3D = go.Layout(
			title=plot_naming + ' - 3D particle - velocity',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# part_h
		layout_part_h_3D = go.Layout(
			title=plot_naming + ' - 3D particle - depth',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)
		
		# part_V 
		layout_part_V_3D = go.Layout(
			title=plot_naming + ' - 3D particle - volume',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)

		# part_P
		layout_part_P_3D = go.Layout(
			title=plot_naming + ' - 3D particle - pressure',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
		)
		
		###############################
		## plot 
		###############################

		# figure
		fig_part_path_3D = go.Figure(data=data_part_path_3D, layout=layout_part_path_3D)
		fig_part_u_3D = go.Figure(data=data_part_u_3D, layout=layout_part_u_3D)
		fig_part_h_3D = go.Figure(data=data_part_h_3D, layout=layout_part_h_3D)
		fig_part_V_3D = go.Figure(data=data_part_V_3D, layout=layout_part_V_3D)
		fig_part_P_3D = go.Figure(data=data_part_P_3D, layout=layout_part_P_3D)

		# plot into html
		plot(fig_part_path_3D, filename=folder_path+plot_naming+' - 3D particle - path.html', auto_open=open_html)
		plot(fig_part_u_3D, filename=folder_path+plot_naming+' - 3D particle - velocity.html', auto_open=open_html)
		plot(fig_part_h_3D, filename=folder_path+plot_naming+' - 3D particle - depth.html', auto_open=open_html)
		plot(fig_part_V_3D, filename=folder_path+plot_naming+' - 3D particle - volume.html', auto_open=open_html)
		plot(fig_part_P_3D, filename=folder_path+plot_naming+' - 3D particle - pressure.html', auto_open=open_html)

		del data_part_path_3D
		del data_part_u_3D
		del data_part_h_3D
		del data_part_P_3D
		del data_part_V_3D
	
	##############################################################
	## flowpath particle + cluster map
	##############################################################
	if cluster_data != None and part_data != None:
		

		###############################
		## particle data
		###############################
		# road
		data_part_cluster_path = [topoSurface]

		# iterate through cluster data
		for cluster_num in part_cID_list: 

			###############################
			## SPEC-debris - particles
			###############################
			# 'pID0','cID1','t2','s3','x4','y5','z6','elevation7','u8','ux9','uy10','h11','V12','P13','i14','j15'

			# select all particles at t == part_t and cluster_id == cluster_num
			# numpy array form for the current flowpath cluster
			part_array_cIDi = np.array(part_df[(part_df['cID'] == cluster_num)])
			
			# x,y-coordinates
			part_x = part_array_cIDi[:,4] 
			part_y = part_array_cIDi[:,5] 
			part_z_plot = part_df_array[:,6] + z_offset 
			# part_z_plot = part_df_array[:,7]

			# other data
			part_pID = part_df_array[:,0].tolist()
			part_cID = part_df_array[:,1].tolist()
			part_t = part_df_array[:,2].tolist()
			part_s = part_df_array[:,3].tolist()
			part_Z = part_df_array[:,6].tolist()
			part_el = part_df_array[:,7].tolist()
			part_ui = part_df_array[:,8].tolist()
			part_uxi = part_df_array[:,9].tolist()
			part_uyi = part_df_array[:,10].tolist()
			part_hi = part_df_array[:,11].tolist()
			part_Vi = part_df_array[:,12].tolist() 			
			part_Pi = part_df_array[:,13].tolist() 			

			###############################
			## particle XY
			###############################
			hovertext_part_path_3D = [(int(pp),int(cc),round(tt,2),round(ss,2),round(zz,2),round(ell,2),round(uu,3),round(uxx,3),round(uyy,3),round(hh,2),round(VV,2),round(PP,2)) for pp,cc,tt,ss,zz,ell,uu,uxx,uyy,hh,VV,PP in zip(part_pID, part_cID, part_t, part_s, part_Z, part_el, part_ui, part_uxi, part_uyi, part_hi, part_Vi, part_Pi)]
			plot_part_xy = go.Scatter3d(
				x=part_x,
				y=part_y,
				z=part_z_plot,
				name='SPEC-debris-cID: '+str(cluster_num),
				hovertext=hovertext_part_path_3D,
				mode='markers',
				marker=dict(
					size=marker_size,
					color=color_list[int(cluster_num)]
				)
			)
			data_part_cluster_path.append(plot_part_xy)

			###############################
			## SPEC-debris - clusters
			###############################
			# 'cID,t,s,x,y,z,u,h,V,D,P,A,Fr,CCHa,merged'

			# select all particles at t == part_t
			# numpy array form for the current flowpath cluster
			# cluster_array_ti_cIDi = np.array(cluster_df[(cluster_df['t'] <= part_t and cluster_df['cID'] == cluster_num)])
			cluster_array_cIDi = np.array(cluster_df[(cluster_df['cID'] == cluster_num)])
			
			# x,y-coordinates
			cluster_x = cluster_array_cIDi[:,3] 
			cluster_y = cluster_array_cIDi[:,4] 
			cluster_z_plot = cluster_array_cIDi[:,5] + z_offset

			# other data
			cluster_t = cluster_array_cIDi[:,1].tolist()
			cluster_s = cluster_array_cIDi[:,2].tolist()
			cluster_Z = cluster_array_cIDi[:,5].tolist()
			cluster_ui = cluster_array_cIDi[:,6].tolist()
			cluster_hi = cluster_array_cIDi[:,7].tolist()
			cluster_Vi = cluster_array_cIDi[:,8].tolist()
			cluster_Di = cluster_array_cIDi[:,9].tolist()
			cluster_Pi = cluster_array_cIDi[:,10].tolist()

			###############################
			## data_path_flowpath 
			###############################
			hovertext_cluster_flowpath_3D = [(round(tt,2),round(cluster_num),round(ss,2),round(zz,2),round(uu,2),round(hh,2),round(VV,2),round(DD,2),round(PP,2)) for tt,ss,zz,uu,hh,VV,DD,PP in zip(cluster_t, cluster_s, cluster_Z, cluster_ui, cluster_hi, cluster_Vi, cluster_Di, cluster_Pi)]
			cluster_plot_xy = go.Scatter3d(
				x=cluster_x,
				y=cluster_y,
				z=cluster_z_plot,
				name='SPEC-debris-flowpath-cluster'+str(cluster_num), 
				hovertext=hovertext_cluster_flowpath_3D,
				mode='lines+markers',
				line=dict(
					width=line_width,
					color=color_list[loop_cID] 
				),
				marker=dict(
					size=marker_size*2,
					color=color_list[loop_cID]
				)
			)
			data_part_cluster_path.append(cluster_plot_xy) 
		
		###############################
		## layout
		###############################
		# part_path
		layout_part_cluster_path = go.Layout(
			title=plot_naming + ' - 3D particle cluster - paths',
			paper_bgcolor="rgba(255,255,255,1)",
			autosize=True,
			width=layout_width, #Nx, #800,
			height=layout_height, #Ny, #1000,
			xaxis=dict(
				# scaleanchor="y",
				# scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='x [m]',
				constrain="domain",
				range=[min(gridUniqueX), max(gridUniqueX)]
			),
			yaxis=dict(
				scaleanchor="x",
				scaleratio=1,
				showline=True,
				linewidth=1, 
				linecolor='black',
				mirror=True,
				autorange=False,
				title='y [m]',
				# constrain="domain",
				range=[min(gridUniqueY), max(gridUniqueY)] 
			),
			showlegend=False, #True,
			legend=dict(orientation="h"),
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			)
			# annotations = dict(
			#     text=plot_annotate,
			#     align='left',
			#     xref='paper',
			#     yref='paper',
			#     x=1.0, 
			#     y=0.1,
			#     showarrow=False
			# )
		)
		
		###############################
		## plot 
		###############################

		# figure
		fig_part_cluster_path = go.Figure(data=data_part_cluster_path, layout=layout_part_cluster_path)

		# plot into html
		plot(fig_part_cluster_path, filename=folder_path+plot_naming+' - 3D particle cluster - path.html', auto_open=open_html)

		del data_part_cluster_path

	return None

###########################################################################
## 2D and 3D plotly animation - particle + cluster - interactive HTML
###########################################################################
def plot_SPEC_debris_animation_2D_plotly_v6_0(folder_path, flowpath_file_name, part_data, cluster_data, cluster_boundary_poly, plot_naming, wall_dict, plot_animation_2D_boundary, animation_duration=100, animation_transition=100, contour_diff=5, max_limits=[20, 10, 2_000, 1000, 2_000, 20, 10, 10, 1000], open_html=True, marker_size=5, line_width=2, layout_width=1000, layout_height=1000):

	##############################################################
	## prefined features
	##############################################################

	if plot_naming == None:
		plot_naming = 'SPEC-debris'

	# black, green, blue, magenta, red, yellow, cyan
	color_list = ['rgba(0, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 0, 255, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 1)', 'rgba(0, 255, 255, 1)']
	color_list_ref = ['rgba(0, 0, 0, 0.75)', 'rgba(0, 255, 0, 0.75)', 'rgba(0, 0, 255, 0.75)', 'rgba(255, 0, 255, 0.75)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 0.75)', 'rgba(0, 255, 255, 0.75)']

	max_resolution = [2000, 5, 0.5, 50, 400]  # volume, velocity, depth, distance, pressure
	# plot_2D_max_limits - cluster u, h, V, D, P, particle u, h, V, P

	colorscale_data= [[0.0, 'rgba(0,0,0,1)'], [0.1, 'rgba(155,0,155,1)'], [0.2, 'rgba(255,0,255,1)'], [0.3, 'rgba(146,0,255,1)'], [0.4, 'rgba(0,0,255,1)'], [0.5, 'rgba(0,104,255,1)'], [0.6, 'rgba(0,255,220,1)'], [0.7, 'rgba(0,255,0,1)'], [0.8, 'rgba(255,255,0,1)'], [0.9, 'rgba(255,130,0,1)'], [1.0, 'rgba(255,0,0,1)']]

	# countour_scale = [[0.0, 'rgba(255,255,255,1)'], [0.1, 'rgba(255,255,255,1)'], [0.2, 'rgba(255,255,255,1)'], [0.3, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,1)'], [0.5, 'rgba(255,255,255,1)'], [0.6, 'rgba(255,255,255,1)'], [0.7, 'rgba(255,255,255,1)'], [0.8, 'rgba(255,255,255,1)'], [0.9, 'rgba(255,255,255,1)'], [1.0, 'rgba(255,255,255,1)']]

	countour_scale = [[0.0, 'rgba(255,255,255,0.0)'], [0.1, 'rgba(255,255,255,0.0)'], [0.2, 'rgba(255,255,255,0.0)'], [0.3, 'rgba(255,255,255,0.0)'], [0.4, 'rgba(255,255,255,0.0)'], [0.5, 'rgba(255,255,255,0.0)'], [0.6, 'rgba(255,255,255,0.0)'], [0.7, 'rgba(255,255,255,0.0)'], [0.8, 'rgba(255,255,255,0.0)'], [0.9, 'rgba(255,255,255,0.0)'], [1.0, 'rgba(255,255,255,0.0)']]
  

	##############################################################
	## 2D contour map
	##############################################################

	DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True)

	topoContour = go.Contour(
		x=gridUniqueX,
		y=gridUniqueY,
		z=DEM,
		line=dict(smoothing=0.85),
		autocontour=False, 
		colorscale=countour_scale,
		showscale = False,
		contours=dict(
			showlabels = True,
			labelfont = dict(
				family = 'Raleway',
				size = 12,
				color = 'black'
			)
			#cmax=np.ceil(max(flowpath.transpose()[loop])),
			#cmin=np.floor(min(flowpath.transpose()[loop]))
		)
	)
	plot_zmax = float(np.ceil(np.max(DEM) + max(max_limits[1], max_limits[6])))

	##############################################################
	## contour path from matplotlib
	##############################################################

	# ref
	# https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_levels.html
	# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html#examples-using-matplotlib-pyplot-contour
	# https://stackoverflow.com/questions/5666056/matplotlib-extracting-data-from-contour-lines

	# number of bins so that elevation contour = 5m
	nn = round((DEM.max() - DEM.min())/contour_diff)+1

	# contour elevation values
	levels = MaxNLocator(nbins=nn).tick_values(DEM.min(), DEM.max())
	
	# contour locations for each elevation values
	# cs = plt.contour(gridUniqueX, gridUniqueY, DEM, levels[1:-1]) 
	cs = plt.contour(gridUniqueX, gridUniqueY, DEM, levels) 
	# plt.show()
	
	## flowpath data
	data_path = []
	# data_path_contour = []
	# data_path=[topoContour] #, goal_plot]
	
	# contour xy
	# data_refresh = [] # [goal_plot]
	# contour_xy = {} # key = z, value = array([x,y])
	# for idx,z_cs in enumerate(levels[1:-1]):
	levels_l = levels.tolist()
	# contour_levels = [levels_l[1]] + levels_l[1:-1]
	# contour_levels = levels_l[1:-1]
	contour_levels = deepcopy(levels_l)

	for idx,z_cs in enumerate(contour_levels):
		# temp_list = []

		# if idx > 0:
		# 	idx_i = idx-1
		# elif idx == 0:
		# 	idx_i = idx

		idx_i = idx

		for loop in range(len(cs.collections[idx_i].get_paths())):
			
			p = cs.collections[idx_i].get_paths()[loop]
			v = p.vertices
			# temp_list.append(v)

			if loop == 0:
				topoContour_i = go.Scatter(
					x=v[:,0],
					y=v[:,1],
					name='Z: '+str(int(z_cs)),
					mode='lines+text',
					line=dict(
						smoothing=0.85,
						width=1,
						color='rgba(0, 0, 0, 0.5)'  # grey
					),
					text=[str(int(z_cs))+'    '],
					textposition="middle center"
				)
			else:
				topoContour_i = go.Scatter(
					x=v[:,0],
					y=v[:,1],
					name='Z: '+str(int(z_cs)),
					mode='lines',
					line=dict(
						smoothing=0.85,
						width=1,
						color='rgba(0, 0, 0, 0.5)'  # grey
					),
				)
			# data_refresh.append(topoContour_i)
			data_path.append(topoContour_i)		
			# data_path_contour.append(topoContour_i)	

		# contour_xy[z_cs] = deepcopy(temp_list)
		# del temp_list


	##############################################################
	## wall_data
	##############################################################
	
	if wall_dict != None:

		# wall_info = [slit_ratio, wall_segment_number, P_or_V ('P' or 'V'), wall_segment_oriP (-90 ~ 90), wall_oriP (-90 ~ 90), thickness, length, Z_opt (1~4), h_or_z, central_X_coord, central_Y_coord]
		# wall_dict -> key = wall_id, value = [(overall) wall_info, [each wall section data], [each wall section shapely polygon]]

		for wall_id in wall_dict.keys():
			for num, wall_seg_poly in enumerate(wall_dict[wall_id][2]):
				corner_pts = np.array(list(wall_seg_poly.exterior.coords))
				wall_seg_i = go.Scatter(
						x=corner_pts[:,0],
						y=corner_pts[:,1],
						name='wall: '+str(wall_id)+str(num),
						mode='lines',
						line=dict(
							width=line_width,
							color='rgba(255, 0, 0, 1)'  # red
						)
					)
				data_path.append(wall_seg_i)

	##############################################################
	## plotly initial data
	##############################################################
	# data_path = []

	fig_map_particle = go.Figure(
		data=data_path
	)

	fig_map_particle_el = go.Figure(
		data=data_path
	)

	fig_map_particle_u = go.Figure(
		data=data_path
	)

	fig_map_particle_h = go.Figure(
		data=data_path
	)

	fig_map_particle_V = go.Figure(
		data=data_path
	)

	fig_map_particle_P = go.Figure(
		data=data_path
	)

	##############################################################
	## dataframe of all particles data
	##############################################################

	# dataframe of all particles data
	part_df_list = []
	pID_list = np.arange(len(part_data[0])).tolist()
	pID_col_array = np.transpose(np.array([pID_list]))

	for part_t in range(len(part_data)):

		# 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P' -> 'pID,cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'

		# numpy array form for the current flowpath cluster
		part_array_ti = np.array(part_data[part_t])
		
		# stack part_num to the part_array_i
		part_array_ii = np.hstack((pID_col_array, part_array_ti))

		# dataframe for each particle
		part_df = pd.DataFrame(part_array_ii, columns=['pID','cID','t','s','x','y','z','elevation','u','ux','uy','h','V','P'])
		part_df_list.append(part_df)
	part_df = pd.concat(part_df_list)

	# total simulation time-step
	part_t_all_list = part_df['t'].values.tolist()
	part_t_list = np.unique(part_t_all_list).tolist()
	part_cID_t0_list = np.unique(part_df['cID'].values.tolist()).tolist()
	part_df_array = part_df.to_numpy()

	##############################################################
	## dataframe of all cluster data
	##############################################################
	if cluster_data != None:
		
		cluster_df_list = []
		for cluster_num in range(len(cluster_data)):
			# 'cID,t,s,x,y,z,u,h,V,D,P,merged'
			# numpy array form for the current flowpath cluster
			cluster_array_i = np.array(cluster_data[int(cluster_num)])
			
			# dataframe for each particle
			cluster_df = pd.DataFrame(cluster_array_i, columns=['cID','t','s','x','y','z','u','h','V','D','P','A','Fr','CCHa','merged'])
			cluster_df_list.append(cluster_df)
		cluster_df = pd.concat(cluster_df_list)

	##############################################################
	## particles animation - frames
	##############################################################

	# add frames and slider information for animation
	frames_path_list = [] 
	frames_el_list = []
	frames_u_list = []
	frames_h_list = []
	frames_V_list = []
	frames_P_list = []
	slider_step_list = []
	for part_time_step, part_time in enumerate(part_t_list):

		frames_data_path_list = []
		frames_data_el_list = []
		frames_data_u_list = []
		frames_data_h_list = []
		frames_data_V_list = []
		frames_data_P_list = []

		# frames_data_path_list = deepcopy(data_path_contour)
		# frames_data_u_list = deepcopy(data_path_contour)
		# frames_data_h_list = deepcopy(data_path_contour)
		# frames_data_V_list = deepcopy(data_path_contour)
		# frames_data_P_list = deepcopy(data_path_contour)
		
		part_array_ti_tt = part_df[(part_df['t'] == part_time)]
		part_cID_list = np.unique(part_array_ti_tt['cID'].values.tolist()).tolist()

		for cluster_num in part_cID_t0_list: 

			if cluster_num in part_cID_list: 

				cluster_num = int(cluster_num)

				###############################
				## SPEC-debris - particles
				###############################
				# 'pID0','cID1','t2','s3','x4','y5','z6','elevation7','u8','ux9','uy10','h11','V12','P13','i14','j15'

				# select all particles at t == part_t and cluster_id == cluster_num
				# numpy array form for the current flowpath cluster
				# part_array_ti_cIDi_tt = part_df[(part_df['t'] == part_time)]
				# part_array_ti_cIDi = np.array(part_array_ti_cIDi_tt[(part_array_ti_cIDi_tt['cID'] == cluster_num)])
				
				part_array_ti_cIDi = np.array(part_array_ti_tt[(part_array_ti_tt['cID'] == cluster_num)])

				# if the does not exist anymore - due to merging
				# if part_array_ti_cIDi.size == 0:
					# continue

				# x,y-coordinates
				part_x = part_array_ti_cIDi[:,4] 
				part_y = part_array_ti_cIDi[:,5] 

				# other data
				part_pID = part_array_ti_cIDi[:,0].tolist()
				part_cID = part_array_ti_cIDi[:,1].tolist()
				part_s = part_array_ti_cIDi[:,3].tolist()
				part_Z = part_array_ti_cIDi[:,6].tolist()
				part_el = part_array_ti_cIDi[:,7].tolist()
				part_ui = part_array_ti_cIDi[:,8].tolist()
				part_uxi = part_array_ti_cIDi[:,9].tolist()
				part_uyi = part_array_ti_cIDi[:,10].tolist()
				part_hi = part_array_ti_cIDi[:,11].tolist()
				part_Vi = part_array_ti_cIDi[:,12].tolist() 
				part_Pi = part_array_ti_cIDi[:,13].tolist() 

				###############################
				## SPEC-debris - particles - flowpath
				###############################
				hovertext_part_path = [(int(pp),int(cc),round(ss,2),round(zz,2),round(ell,2),round(uu,3),round(uxx,3),round(uyy,3),round(hh,2),round(VV,2),round(PP,2)) for pp,cc,ss,zz,ell,uu,uxx,uyy,hh,VV,PP in zip(part_pID, part_cID, part_s, part_Z, part_el, part_ui, part_uxi, part_uyi, part_hi, part_Vi, part_Pi)]
				plot_part_xy = go.Scatter(
					x=part_x,
					y=part_y,
					name='SPEC-debris-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_path,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=color_list[int(cluster_num)]
					)
				)

				###############################
				## SPEC-debris - particles - elevation
				###############################
				hovertext_part_el = [(round(zz,2),round(ell,2)) for zz,ell in zip(part_Z, part_el)]
				plot_part_el = go.Scatter(
					x=part_x,
					y=part_y,
					name='SPEC-debris-elevation-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_el,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_el,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='elevation [m]',
							ticks="outside",
						),
						cmax=plot_zmax,
						cmin=0
					)
				)

				###############################
				## SPEC-debris - particles - velocity
				###############################
				# hovertext_part_u = [(int(pp),int(cc),round(ss,2),round(zz,2),round(uu,3),round(uxx,3),round(uyy,3)) for pp,cc,ss,zz,uu,uxx,uyy in zip(part_pID, part_cID, part_s, part_Z, part_ui, part_uxi, part_uyi)]
				hovertext_part_u = [(round(uu,3),round(uxx,3),round(uyy,3)) for uu,uxx,uyy in zip(part_ui, part_uxi, part_uyi)]
				plot_part_u = go.Scatter(
					x=part_x,
					y=part_y,
					name='SPEC-debris-velocity-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_u,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_ui,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='velocity [m/s]',
							ticks="outside",
						),
						cmax=max_limits[5],
						cmin=0
					)
				)

				###############################
				## SPEC-debris - particles - depth
				###############################
				# hovertext_part_h = [(int(pp),int(cc),round(ss,2),round(zz,2),round(hh,2)) for pp,cc,ss,zz,hh in zip(part_pID, part_cID, part_s, part_Z, part_hi)]
				hovertext_part_h = np.round(part_hi,decimals=2)
				plot_part_h = go.Scatter(
					x=part_x,
					y=part_y,
					name='SPEC-debris-depth-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_h,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_hi,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='depth [m]',
							ticks="outside",
						),
						cmax=max_limits[6],
						cmin=0
					)
				)

				###############################
				## SPEC-debris - particles - volume
				###############################
				# hovertext_part_V = [(int(pp),int(cc),round(ss,2),round(zz,2),round(VV,2)) for pp,cc,ss,zz,VV in zip(part_pID, part_cID, part_s, part_Z, part_Vi)]
				hovertext_part_V = np.round(part_Vi,decimals=2)
				plot_part_V = go.Scatter(
					x=part_x,
					y=part_y,
					name='SPEC-debris-volume-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_V,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_Vi,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='volume [m^3]',
							ticks="outside",
						),
						cmax=max_limits[7],
						cmin=0
					)
				)

				###############################
				## SPEC-debris - particles - pressure
				###############################
				# hovertext_part_P = [(int(pp),int(cc),round(ss,2),round(zz,2),round(PP,2)) for pp,cc,ss,zz,PP in zip(part_pID, part_cID, part_s, part_Z, part_Pi)]
				hovertext_part_P = np.round(part_Pi,decimals=2)
				plot_part_P = go.Scatter(
					x=part_x,
					y=part_y,
					name='SPEC-debris-pressure-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_P,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_Pi,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='pressure [kPa]',
							ticks="outside",
						),
						cmax=max_limits[8],
						cmin=0
					)
				)

			# create empty ones to keep consistancy in animation data - plotly animation plot issue
			else: 
				cluster_num = int(cluster_num)

				###############################
				## SPEC-debris - particles - flowpath
				###############################
				plot_part_xy = go.Scatter(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					name='SPEC-debris-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=marker_size,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - elevation
				###############################
				plot_part_el = go.Scatter(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					name='SPEC-debris-elevation-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=marker_size,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - velocity
				###############################
				plot_part_u = go.Scatter(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					name='SPEC-debris-velocity-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=marker_size,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - depth
				###############################
				plot_part_h = go.Scatter(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					name='SPEC-debris-depth-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=marker_size,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - volume
				###############################
				plot_part_V = go.Scatter(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					name='SPEC-debris-volume-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=marker_size,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - pressure
				###############################
				plot_part_P = go.Scatter(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					name='SPEC-debris-pressure-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=marker_size,
						color='rgba(255,255,255,0.0)'
					)
				)

			frames_data_path_list.append(plot_part_xy)
			frames_data_el_list.append(plot_part_el)
			frames_data_u_list.append(plot_part_u)
			frames_data_h_list.append(plot_part_h)
			frames_data_V_list.append(plot_part_V)
			frames_data_P_list.append(plot_part_P)

		###############################
		## SPEC-debris - cluster data
		###############################
		if cluster_data != None:

			for cluster_num in part_cID_t0_list:

				###############################
				## SPEC-debris - clusters
				###############################
				# 'cID,t,s,x,y,z,u,h,V,D,P,A,CCHa,merged'

				# select all particles at t == part_t
				# numpy array form for the current flowpath cluster
				# cluster_array_ti_cIDi = np.array(cluster_df[(cluster_df['t'] <= part_t and cluster_df['cID'] == cluster_num)])
				# cluster_array_ti_cIDi_tt = cluster_df[(cluster_df['t'] <= part_time)]
				# cluster_array_ti_cIDi = np.array(cluster_array_ti_cIDi_tt[(cluster_array_ti_cIDi_tt['cID'] == cluster_num)])

				cluster_array_ti_cIDi = np.array(cluster_data[int(cluster_num)][:(part_time_step+1)])

				# if merging occurred, no longer plot them
				# if cluster_array_ti_cIDi[0,-1] == 1:
					# continue 
				
				# x,y-coordinates
				cluster_x = cluster_array_ti_cIDi[:,3] 
				cluster_y = cluster_array_ti_cIDi[:,4] 

				# other data
				cluster_s = cluster_array_ti_cIDi[:,2].tolist()
				cluster_Z = cluster_array_ti_cIDi[:,5].tolist()
				cluster_ui = cluster_array_ti_cIDi[:,6].tolist()
				cluster_hi = cluster_array_ti_cIDi[:,7].tolist()
				cluster_Vi = cluster_array_ti_cIDi[:,8].tolist()
				cluster_Di = cluster_array_ti_cIDi[:,9].tolist()
				cluster_Pi = cluster_array_ti_cIDi[:,10].tolist()

				###############################
				## data_path_flowpath 
				###############################
				hovertext_cluster_flowpath = [(round(part_time,2),round(cluster_num),round(ss,2),round(zz,2),round(uu,2),round(hh,2),round(VV,2),round(DD,2),round(PP,2)) for ss,zz,uu,hh,VV,DD,PP in zip(cluster_s, cluster_Z, cluster_ui, cluster_hi, cluster_Vi, cluster_Di, cluster_Pi)]
				cluster_plot_xy = go.Scatter(
					x=cluster_x,
					y=cluster_y,
					name='SPEC-debris-flowpath-cluster'+str(cluster_num), 
					hovertext=hovertext_cluster_flowpath,
					mode='lines+markers',
					line=dict(
						width=line_width,
						color=color_list[int(cluster_num)] 
					),
					marker=dict(
						size=marker_size*2,
						color=color_list[int(cluster_num)]
					)
				)
				frames_data_path_list.append(cluster_plot_xy) 
				frames_data_u_list.append(cluster_plot_xy) 
				frames_data_h_list.append(cluster_plot_xy) 
				frames_data_V_list.append(cluster_plot_xy) 
				frames_data_P_list.append(cluster_plot_xy) 

		###############################
		## cluster boundary polygon 
		###############################
		if cluster_boundary_poly != None and plot_animation_2D_boundary == True:

			for cluster_num_b in part_cID_t0_list:

				if cluster_num_b in part_cID_list:

					# cluster_boundary_poly -> 1st index = cID, 2nd index = time-step
					cluster_time_boundary_polygon = cluster_boundary_poly[int(cluster_num_b)][part_time_step]
					cluster_time_boundary_x, cluster_time_boundary_y = cluster_time_boundary_polygon.exterior.coords.xy
					cluster_time_boundary_x = cluster_time_boundary_x.tolist()
					cluster_time_boundary_y = cluster_time_boundary_y.tolist()

					cluster_boundary_plot = go.Scatter(
						x=cluster_time_boundary_x,
						y=cluster_time_boundary_y,
						name='SPEC-debris-flowpath-cluster-boundary'+str(cluster_num_b)+'t'+str(part_time_step), 
						mode='lines',
						line=dict(
							width=line_width*0.8,
							color=color_list[int(cluster_num_b)] 
						)
					)	

				# create empty ones to keep consistancy in animation data - plotly animation plot issue
				else:
					cluster_boundary_plot = go.Scatter(
						x=[min(gridUniqueX), min(gridUniqueX)+deltaX],
						y=[min(gridUniqueY), min(gridUniqueY)+deltaY],
						name='SPEC-debris-flowpath-cluster-boundary'+str(cluster_num_b)+'t'+str(part_time_step), 
						mode='lines',
						line=dict(
							width=1.0,
							color='rgba(255,255,255,0.0)'
						)
					)

				frames_data_path_list.append(cluster_boundary_plot) 
				frames_data_u_list.append(cluster_boundary_plot) 
				frames_data_h_list.append(cluster_boundary_plot) 
				frames_data_V_list.append(cluster_boundary_plot) 
				frames_data_P_list.append(cluster_boundary_plot)

		###############################
		## add frame and slider results 
		###############################
		# add contour
		# frames_data_path_list.append(topoContour)	
		# frames_data_u_list.append(topoContour)	
		# frames_data_h_list.append(topoContour)	
		# frames_data_V_list.append(topoContour)	
		# frames_data_P_list.append(topoContour)	

		frames_path_list.append(go.Frame(data=frames_data_path_list, name=str(part_time_step)))
		frames_el_list.append(go.Frame(data=frames_data_el_list, name=str(part_time_step)))
		frames_u_list.append(go.Frame(data=frames_data_u_list, name=str(part_time_step)))
		frames_h_list.append(go.Frame(data=frames_data_h_list, name=str(part_time_step)))
		frames_V_list.append(go.Frame(data=frames_data_V_list, name=str(part_time_step)))
		frames_P_list.append(go.Frame(data=frames_data_P_list, name=str(part_time_step)))
		
		del frames_data_path_list, frames_data_el_list, frames_data_u_list, frames_data_h_list, frames_data_V_list, frames_data_P_list

		## slide_step information
		slider_step_dict = {
			'args': [ [part_time_step],
				{'frame': {'duration': animation_duration, 'redraw': True},
				# {'frame': {'duration': animation_duration, 'redraw': False},
				'mode': 'immediate',
			   'transition': {'duration': animation_transition}}
			],
			'label': str(round(part_time,3)), # str(part_time_step),
			'method': 'animate'
		}
		slider_step_list.append(slider_step_dict)

	# add contour
	# fig_map_particle.data = [topoContour]	
	# fig_map_particle_u.data = [topoContour]	
	# fig_map_particle_h.data = [topoContour]	
	# fig_map_particle_V.data = [topoContour]	
	# fig_map_particle_P.data = [topoContour]

	fig_map_particle.frames = frames_path_list
	fig_map_particle_el.frames = frames_el_list
	fig_map_particle_u.frames = frames_u_list
	fig_map_particle_h.frames = frames_h_list
	fig_map_particle_V.frames = frames_V_list
	fig_map_particle_P.frames = frames_P_list

	##############################################################
	## layout
	##############################################################

	fig_map_particle.layout = go.Layout(
		title=plot_naming+" - flowpath",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # 
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
					# 'args': [None, {'frame': {'duration': animation_duration, 'redraw': False},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					# 'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
					'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	fig_map_particle_el.layout = go.Layout(
		title=plot_naming+" - elevation",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
					# 'args': [None, {'frame': {'duration': animation_duration, 'redraw': False},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					# 'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
					'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)


	fig_map_particle_u.layout = go.Layout(
		title=plot_naming+" - velocity",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
					# 'args': [None, {'frame': {'duration': animation_duration, 'redraw': False},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					# 'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
					'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	fig_map_particle_h.layout = go.Layout(
		title=plot_naming+" - depth",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
					# 'args': [None, {'frame': {'duration': animation_duration, 'redraw': False},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					# 'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
					'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	fig_map_particle_V.layout = go.Layout(
		title=plot_naming+" - volume",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
					# 'args': [None, {'frame': {'duration': animation_duration, 'redraw': False},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					# 'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
					'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	fig_map_particle_P.layout = go.Layout(
		title=plot_naming+" - pressure",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
					# 'args': [None, {'frame': {'duration': animation_duration, 'redraw': False},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					# 'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
					'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	##############################################################
	## plot 
	##############################################################

	if cluster_data != None:
		plot(fig_map_particle, filename=folder_path+plot_naming+' - 2D-particle-cluster-animation - flowpath.html', auto_open=open_html)
		plot(fig_map_particle_el, filename=folder_path+plot_naming+' - 2D-particle-cluster-animation - elevation.html', auto_open=open_html)
		plot(fig_map_particle_u, filename=folder_path+plot_naming+' - 2D-particle-cluster-animation - velocity.html', auto_open=open_html)
		plot(fig_map_particle_h, filename=folder_path+plot_naming+' - 2D-particle-cluster-animation - depth.html', auto_open=open_html)
		plot(fig_map_particle_V, filename=folder_path+plot_naming+' - 2D-particle-cluster-animation - volume.html', auto_open=open_html)
		plot(fig_map_particle_P, filename=folder_path+plot_naming+' - 2D-particle-cluster-animation - pressure.html', auto_open=open_html)
	else:
		plot(fig_map_particle, filename=folder_path+plot_naming+' - 2D-particle-animation - flowpath.html', auto_open=open_html)
		plot(fig_map_particle_el, filename=folder_path+plot_naming+' - 2D-particle-animation - elevation.html', auto_open=open_html)
		plot(fig_map_particle_u, filename=folder_path+plot_naming+' - 2D-particle-animation - velocity.html', auto_open=open_html)
		plot(fig_map_particle_h, filename=folder_path+plot_naming+' - 2D-particle-animation - depth.html', auto_open=open_html)
		plot(fig_map_particle_V, filename=folder_path+plot_naming+' - 2D-particle-animation - volume.html', auto_open=open_html)
		plot(fig_map_particle_P, filename=folder_path+plot_naming+' - 2D-particle-animation - pressure.html', auto_open=open_html)

	return None

def plot_SPEC_debris_animation_3D_plotly_v5_0(folder_path, flowpath_file_name, part_data, cluster_data, plot_naming, animation_duration=100, animation_transition=100, contour_diff=5, max_limits=[20, 10, 2_000, 1000, 2_000, 20, 10, 10, 1000], open_html=True, z_offset=0, marker_size=5, line_width=2, layout_width=1000, layout_height=1000):

	##############################################################
	## prefined features
	##############################################################

	if plot_naming == None:
		plot_naming = 'SPEC-debris'

	# black, green, blue, magenta, red, yellow, cyan
	color_list = ['rgba(0, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 0, 255, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 1)', 'rgba(0, 255, 255, 1)']
	color_list_ref = ['rgba(0, 0, 0, 0.75)', 'rgba(0, 255, 0, 0.75)', 'rgba(0, 0, 255, 0.75)', 'rgba(255, 0, 255, 0.75)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 0.75)', 'rgba(0, 255, 255, 0.75)']

	max_resolution = [2000, 5, 0.5, 50, 400]  # volume, velocity, depth, distance, pressure
	# plot_2D_max_limits - cluster u, h, V, D, P, particle u, h, V, P

	colorscale_data= [[0.0, 'rgba(0,0,0,1)'], [0.1, 'rgba(155,0,155,1)'], [0.2, 'rgba(255,0,255,1)'], [0.3, 'rgba(146,0,255,1)'], [0.4, 'rgba(0,0,255,1)'], [0.5, 'rgba(0,104,255,1)'], [0.6, 'rgba(0,255,220,1)'], [0.7, 'rgba(0,255,0,1)'], [0.8, 'rgba(255,255,0,1)'], [0.9, 'rgba(255,130,0,1)'], [1.0, 'rgba(255,0,0,1)']]

	countour_scale = [[0.0, 'rgba(255,255,255,1)'], [0.1, 'rgba(255,255,255,1)'], [0.2, 'rgba(255,255,255,1)'], [0.3, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,1)'], [0.5, 'rgba(255,255,255,1)'], [0.6, 'rgba(255,255,255,1)'], [0.7, 'rgba(255,255,255,1)'], [0.8, 'rgba(255,255,255,1)'], [0.9, 'rgba(255,255,255,1)'], [1.0, 'rgba(255,255,255,1)']]

	##############################################################
	## 2D contour map
	##############################################################
	
	DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True)

	topoSurface = go.Surface(
		x=gridUniqueX,
		y=gridUniqueY,
		z=DEM,
		colorscale='geyser',
		showscale = False,
		# contours = {
		# 	"x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color":"black"},
		# 	"z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
		# }
	)

	plot_zmax = float(np.ceil(np.max(DEM) + max(max_limits[1], max_limits[6])))

	##############################################################
	## contour path from matplotlib
	##############################################################
	
	# ref
	# https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_levels.html
	# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html#examples-using-matplotlib-pyplot-contour
	# https://stackoverflow.com/questions/5666056/matplotlib-extracting-data-from-contour-lines

	# number of bins so that elevation contour = 5m
	nn = round((DEM.max() - DEM.min())/contour_diff)+1

	# contour elevation values
	levels = MaxNLocator(nbins=nn).tick_values(DEM.min(), DEM.max())
	
	# contour locations for each elevation values
	cs = plt.contour(gridUniqueX, gridUniqueY, DEM, levels[1:-1]) 
	# plt.show()
	
	## flowpath data
	data_path = []
	
	# contour xy
	levels_l = levels.tolist()
	contour_levels = levels_l[1:-1]

	for idx,z_cs in enumerate(contour_levels):

		idx_i = idx

		for loop in range(len(cs.collections[idx_i].get_paths())):
			
			p = cs.collections[idx_i].get_paths()[loop]
			v = p.vertices
			z_array = z_cs*np.ones(len(v[:,0]))

			if loop == 0:
				topoContour_i = go.Scatter3d(
					x=v[:,0],
					y=v[:,1],
					z=z_array,
					name='Z: '+str(int(z_cs)),
					mode='lines+text',
					line=dict(
						width=1,
						color='rgba(0, 0, 0, 0.75)'  # grey
					),
					text=[str(int(z_cs))+'    '],
					textposition="middle center"
				)
			else:
				topoContour_i = go.Scatter3d(
					x=v[:,0],
					y=v[:,1],
					z=z_array,
					name='Z: '+str(int(z_cs)),
					mode='lines',
					line=dict(
						width=1,
						color='rgba(0, 0, 0, 0.75)'  # grey
					),
				)
			data_path.append(topoContour_i)		


	##############################################################
	## plotly initial data
	##############################################################
	
	data_path.append(topoSurface)
	
	fig_map_particle = go.Figure(
		data=data_path
	)

	fig_map_particle_u = go.Figure(
		data=data_path
	)

	fig_map_particle_h = go.Figure(
		data=data_path
	)

	fig_map_particle_V = go.Figure(
		data=data_path
	)

	fig_map_particle_P = go.Figure(
		data=data_path
	)

	##############################################################
	## dataframe of all particles data
	##############################################################

	# dataframe of all particles data
	part_df_list = []
	pID_list = np.arange(len(part_data[0])).tolist()
	pID_col_array = np.transpose(np.array([pID_list]))

	for part_t in range(len(part_data)):

		# 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P' -> 'pID,cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'

		# numpy array form for the current flowpath cluster
		part_array_ti = np.array(part_data[part_t])
		
		# stack part_num to the part_array_i
		part_array_ii = np.hstack((pID_col_array, part_array_ti))

		# dataframe for each particle
		part_df = pd.DataFrame(part_array_ii, columns=['pID','cID','t','s','x','y','z','elevation','u','ux','uy', 'h', 'V', 'P'])
		part_df_list.append(part_df)
	part_df = pd.concat(part_df_list)

	# total simulation time-step
	part_t_all_list = part_df['t'].values.tolist()
	part_t_list = np.unique(part_t_all_list).tolist()
	part_cID_t0_list = np.unique(part_df['cID'].values.tolist()).tolist()
	part_df_array = part_df.to_numpy()

	##############################################################
	## dataframe of all cluster data
	##############################################################
	if cluster_data != None:
		
		cluster_df_list = []
		for cluster_num in range(len(cluster_data)):
			# 'cID,t,s,x,y,z,u,h,V,D,P,merged'
			# numpy array form for the current flowpath cluster
			cluster_array_i = np.array(cluster_data[int(cluster_num)])
			
			# dataframe for each particle
			cluster_df = pd.DataFrame(cluster_array_i, columns=['cID','t','s','x','y','z','u','h','V','D','P','A','Fr','CCHa','merged'])
			cluster_df_list.append(cluster_df)
		cluster_df = pd.concat(cluster_df_list)

	##############################################################
	## particles animation - frames
	##############################################################

	# add frames and slider information for animation
	frames_path_list = []
	frames_u_list = []
	frames_h_list = []
	frames_V_list = []
	frames_P_list = []
	slider_step_list = []
	for part_time_step, part_time in enumerate(part_t_list):

		frames_data_path_list = []
		frames_data_u_list = []
		frames_data_h_list = []
		frames_data_V_list = []
		frames_data_P_list = []

		part_array_ti_tt = part_df[(part_df['t'] == part_time)]
		part_cID_list = np.unique(part_array_ti_tt['cID'].values.tolist()).tolist()

		for cluster_num in part_cID_t0_list: 

			if cluster_num in part_cID_list:

				cluster_num = int(cluster_num)

				###############################
				## SPEC-debris - particles
				###############################
				# 'pID,cID,t,s,x,y,z,u,ux,uy,h,V,P'
				# 'pID0','cID1','t2','s3','x4','y5','z6','elevation7','u8','ux9','uy10','h11','V12','P13','i14','j15'

				# select all particles at t == part_t and cluster_id == cluster_num
				# numpy array form for the current flowpath cluster
				part_array_ti_cIDi_tt = part_df[(part_df['t'] == part_time)]
				part_array_ti_cIDi = np.array(part_array_ti_cIDi_tt[(part_array_ti_cIDi_tt['cID'] == cluster_num)])
				
				# x,y-coordinates
				part_x = part_array_ti_cIDi[:,4] 
				part_y = part_array_ti_cIDi[:,5] 
				part_z_plot = part_array_ti_cIDi[:,6] + z_offset
				# part_z_plot = part_array_ti_cIDi[:,7]

				# other data
				part_pID = part_array_ti_cIDi[:,0].tolist()
				part_cID = part_array_ti_cIDi[:,1].tolist()
				part_s = part_array_ti_cIDi[:,3].tolist()
				part_Z = part_array_ti_cIDi[:,6].tolist()
				part_el = part_array_ti_cIDi[:,7].tolist()
				part_ui = part_array_ti_cIDi[:,8].tolist()
				part_uxi = part_array_ti_cIDi[:,9].tolist()
				part_uyi = part_array_ti_cIDi[:,10].tolist()
				part_hi = part_array_ti_cIDi[:,11].tolist()
				part_Vi = part_array_ti_cIDi[:,12].tolist() 
				part_Pi = part_array_ti_cIDi[:,13].tolist() 

				###############################
				## SPEC-debris - particles - flowpath
				###############################
				hovertext_part_path = [(int(pp),int(cc),round(ss,2),round(zz,2),round(ell,2),round(uu,3),round(uxx,3),round(uyy,3),round(hh,2),round(VV,2),round(PP,2)) for pp,cc,ss,zz,ell,uu,uxx,uyy,hh,VV,PP in zip(part_pID, part_cID, part_s, part_Z, part_el, part_ui, part_uxi, part_uyi, part_hi, part_Vi, part_Pi)]
				plot_part_xy = go.Scatter3d(
					x=part_x,
					y=part_y,
					z=part_z_plot,
					name='SPEC-debris-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_path,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=color_list[int(cluster_num)]
					)
				)

				###############################
				## SPEC-debris - particles - velocity
				###############################
				hovertext_part_u = [(int(pp),int(cc),round(ss,2),round(zz,2),round(ell,2),round(uu,3),round(uxx,3),round(uyy,3)) for pp,cc,ss,zz,ell,uu,uxx,uyy in zip(part_pID, part_cID, part_s, part_Z, part_el, part_ui, part_uxi, part_uyi)]
				plot_part_u = go.Scatter3d(
					x=part_x,
					y=part_y,
					z=part_z_plot,
					name='SPEC-debris-velocity-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_u,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_ui,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='velocity [m/s]',
							ticks="outside",
						),
						cmax=max_limits[5],
						cmin=0
					)
				)

				###############################
				## SPEC-debris - particles - depth
				###############################
				hovertext_part_h = [(int(pp),int(cc),round(ss,2),round(zz,2),round(ell,2),round(hh,3)) for pp,cc,ss,zz,ell,hh in zip(part_pID, part_cID, part_s, part_Z, part_el, part_hi)]
				plot_part_h = go.Scatter3d(
					x=part_x,
					y=part_y,
					z=part_z_plot,
					name='SPEC-debris-depth-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_h,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_hi,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='depth [m]',
							ticks="outside",
						),
						cmax=max_limits[6],
						cmin=0
					)
				)

				###############################
				## SPEC-debris - particles - volume
				###############################
				hovertext_part_V = [(int(pp),int(cc),round(ss,2),round(zz,2),round(ell,2),round(VV,2)) for pp,cc,ss,zz,ell,VV in zip(part_pID, part_cID, part_s, part_Z, part_el, part_Vi)]
				plot_part_V = go.Scatter3d(
					x=part_x,
					y=part_y,
					z=part_z_plot,
					name='SPEC-debris-volume-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_V,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_Vi,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='volume [m^3]',
							ticks="outside",
						),
						cmax=max_limits[7],
						cmin=0
					)
				)

				###############################
				## SPEC-debris - particles - pressure
				###############################
				hovertext_part_P = [(int(pp),int(cc),round(ss,2),round(zz,2),round(ell,2),round(PP,2)) for pp,cc,ss,zz,ell,PP in zip(part_pID, part_cID, part_s, part_Z, part_el, part_Pi)]
				plot_part_P = go.Scatter3d(
					x=part_x,
					y=part_y,
					z=part_z_plot,
					name='SPEC-debris-pressure-cID: '+str(cluster_num)+' time:'+str(part_time), 
					hovertext=hovertext_part_P,
					mode='markers',
					marker=dict(
						size=marker_size,
						color=np.round(part_Pi,decimals=2),
						colorscale=colorscale_data,
						colorbar=dict(
							title='pressure [kPa]',
							ticks="outside",
						),
						cmax=max_limits[8],
						cmin=0
					)
				)

			else:

				###############################
				## SPEC-debris - particles - flowpath
				###############################
				plot_part_xy = go.Scatter3d(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					z=[0.0],
					name='SPEC-debris-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=1.0,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - velocity
				###############################
				plot_part_u = go.Scatter3d(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					z=[0.0],
					name='SPEC-debris-velocity-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=1.0,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - depth
				###############################
				plot_part_h = go.Scatter3d(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					z=[0.0],
					name='SPEC-debris-depth-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=1.0,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - volume
				###############################
				plot_part_V = go.Scatter3d(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					z=[0.0],
					name='SPEC-debris-volume-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=1.0,
						color='rgba(255,255,255,0.0)'
					)
				)

				###############################
				## SPEC-debris - particles - pressure
				###############################
				plot_part_P = go.Scatter3d(
					x=[min(gridUniqueX)],
					y=[min(gridUniqueY)],
					z=[0.0],
					name='SPEC-debris-pressure-cID: '+str(cluster_num)+' time:'+str(part_time), 
					mode='markers',
					marker=dict(
						size=1.0,
						color='rgba(255,255,255,0.0)'
					)
				)

			frames_data_path_list.append(plot_part_xy)
			frames_data_u_list.append(plot_part_u)
			frames_data_h_list.append(plot_part_h)
			frames_data_V_list.append(plot_part_V)
			frames_data_P_list.append(plot_part_P)

			###############################
			## SPEC-debris - cluster data
			###############################
			if cluster_data != None:

				###############################
				## SPEC-debris - clusters
				###############################
				# 'cID,t,s,x,y,z,u,h,V,D,P,merged'

				# select all particles at t == part_t
				# numpy array form for the current flowpath cluster
				# cluster_array_ti_cIDi = np.array(cluster_df[(cluster_df['t'] <= part_t and cluster_df['cID'] == cluster_num)])
				cluster_array_ti_cIDi_tt = cluster_df[(cluster_df['t'] <= part_time)]
				cluster_array_ti_cIDi = np.array(cluster_array_ti_cIDi_tt[(cluster_array_ti_cIDi_tt['cID'] == cluster_num)])
				
				# x,y-coordinates
				cluster_x = cluster_array_ti_cIDi[:,3] 
				cluster_y = cluster_array_ti_cIDi[:,4] 
				cluster_z_plot = cluster_array_ti_cIDi[:,5] + z_offset

				# other data
				cluster_s = cluster_array_ti_cIDi[:,2].tolist()
				cluster_Z = cluster_array_ti_cIDi[:,5].tolist()
				cluster_ui = cluster_array_ti_cIDi[:,6].tolist()
				cluster_hi = cluster_array_ti_cIDi[:,7].tolist()
				cluster_Vi = cluster_array_ti_cIDi[:,8].tolist()
				cluster_Di = cluster_array_ti_cIDi[:,9].tolist()
				cluster_Pi = cluster_array_ti_cIDi[:,10].tolist()

				###############################
				## data_path_flowpath 
				###############################
				hovertext_cluster_flowpath = [(round(part_time,2),round(cluster_num),round(ss,2),round(zz,2),round(uu,2),round(hh,2),round(VV,2),round(DD,2),round(PP,2)) for ss,zz,uu,hh,VV,DD,PP in zip(cluster_s, cluster_Z, cluster_ui, cluster_hi, cluster_Vi, cluster_Di, cluster_Pi)]
				cluster_plot_xy = go.Scatter3d(
					x=cluster_x,
					y=cluster_y,
					z=cluster_z_plot,
					name='SPEC-debris-flowpath-cluster'+str(cluster_num), 
					hovertext=hovertext_cluster_flowpath,
					mode='lines+markers',
					line=dict(
						width=line_width,
						color=color_list[int(cluster_num)] 
					),
					marker=dict(
						size=marker_size*0.5,
						color=color_list[int(cluster_num)]
					)
				)
				frames_data_path_list.append(cluster_plot_xy) 
				frames_data_u_list.append(cluster_plot_xy) 
				frames_data_h_list.append(cluster_plot_xy) 
				frames_data_V_list.append(cluster_plot_xy) 
				frames_data_P_list.append(cluster_plot_xy) 
			
		frames_path_list.append(go.Frame(data=frames_data_path_list, name=str(part_time_step)))
		frames_u_list.append(go.Frame(data=frames_data_u_list, name=str(part_time_step)))
		frames_h_list.append(go.Frame(data=frames_data_h_list, name=str(part_time_step)))
		frames_V_list.append(go.Frame(data=frames_data_V_list, name=str(part_time_step)))
		frames_P_list.append(go.Frame(data=frames_data_P_list, name=str(part_time_step)))
		
		del frames_data_path_list, frames_data_u_list, frames_data_h_list, frames_data_V_list, frames_data_P_list
		
		## slide_step information
		slider_step_dict = {
			'args': [ [part_time_step],
				{'frame': {'duration': animation_duration, 'redraw': True},
				'mode': 'immediate',
			   'transition': {'duration': animation_transition}}
			],
			'label': str(round(part_time,3)), # str(part_time_step),
			'method': 'animate'
		}
		slider_step_list.append(slider_step_dict)

	fig_map_particle.frames = frames_path_list
	fig_map_particle_u.frames = frames_u_list
	fig_map_particle_h.frames = frames_h_list
	fig_map_particle_V.frames = frames_V_list
	fig_map_particle_P.frames = frames_P_list

	##############################################################
	## layout
	##############################################################

	fig_map_particle.layout = go.Layout(
		title=plot_naming+" - flowpath",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	fig_map_particle_u.layout = go.Layout(
		title=plot_naming+" - velocity",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	fig_map_particle_h.layout = go.Layout(
		title=plot_naming+" - depth",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	fig_map_particle_V.layout = go.Layout(
		title=plot_naming+" - volume",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	fig_map_particle_P.layout = go.Layout(
		title=plot_naming+" - pressure",
		autosize=True,
		paper_bgcolor='rgba(255,255,255,1)',
		plot_bgcolor='rgba(255,255,255,1)',
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=False,  # True, #
		legend=dict(orientation="v"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		updatemenus=[dict(
			buttons=[
				{
					'args': [None, {'frame': {'duration': animation_duration, 'redraw': True},
							 'fromcurrent': True, 'mode': 'immediate',
							 'transition': {'duration': animation_transition, 'easing': 'quadratic-in-out'}}],
					'label': 'Play',
					'method': 'animate'
				},
				{
					'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
							'transition': {'duration': 0}}],
					'label': 'Pause',
					'method': 'animate'
				}
			],
			direction='left',
			pad={'r': 10, 't': 87},
			showactive=False,
			type='buttons',
			x=0.1,
			xanchor='right',
			y=0,
			yanchor='top'
		)],
		sliders = [dict(
			active= 0,
			yanchor='top',
			xanchor='left',
			currentvalue=dict(
				font=dict(size=20),
				# prefix='time-step: ',
				prefix='time(s): ',
				visible=True,
				xanchor='right'
			),
			transition=dict(duration=animation_transition, easing='cubic-in-out'),
			pad=dict(b=10, t=50),
			len=0.9,
			x=0.1,
			y=0,
			steps=slider_step_list
		)]
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	## figure plotting
	# fig_map_particle = go.Figure(
	# 	{'data': data_path,
	# 	'layout': layout_path_part,
	# 	'frames': frames_list
	# 	}
	# )

	# fig_map_particle.show()

	##############################################################
	## plot 
	##############################################################

	if cluster_data != None:
		plot(fig_map_particle, filename=folder_path+plot_naming+' - 3D-particle-cluster-animation - flowpath.html', auto_open=open_html)
		plot(fig_map_particle_u, filename=folder_path+plot_naming+' - 3D-particle-cluster-animation - velocity.html', auto_open=open_html)
		plot(fig_map_particle_h, filename=folder_path+plot_naming+' - 3D-particle-cluster-animation - depth.html', auto_open=open_html)
		plot(fig_map_particle_V, filename=folder_path+plot_naming+' - 3D-particle-cluster-animation - volume.html', auto_open=open_html)
		plot(fig_map_particle_P, filename=folder_path+plot_naming+' - 3D-particle-cluster-animation - pressure.html', auto_open=open_html)
	else:
		plot(fig_map_particle, filename=folder_path+plot_naming+' - 3D-particle-animation - flowpath.html', auto_open=open_html)
		plot(fig_map_particle_u, filename=folder_path+plot_naming+' - 3D-particle-animation - velocity.html', auto_open=open_html)
		plot(fig_map_particle_h, filename=folder_path+plot_naming+' - 3D-particle-animation - depth.html', auto_open=open_html)
		plot(fig_map_particle_V, filename=folder_path+plot_naming+' - 3D-particle-animation - volume.html', auto_open=open_html)
		plot(fig_map_particle_P, filename=folder_path+plot_naming+' - 3D-particle-animation - pressure.html', auto_open=open_html)

	return None

###########################################################################
## 2D plotly interactive map - HTML - optimal barrier location
###########################################################################
def plot_flowpath_network_closed_v2_1(folder_path, flowpath_file_name, road_xy_list, flowpath_dfl, flowpath_link, parameter_at_dfl, plot_naming, dp=2, marker_size=5, line_width=2, layout_width=1000, layout_height=1000):

	'''
	flowpath_link 		# key = cluster id, value = [(0) [linking dfl_ids], (1) [dfl_type]]
	flowpath_dfl  		# key = dfl id, value = [(0) dfl_type, (1) (x,y), (2) [cluster class]]
	parameter_at_dfl	# key = dfl, value = [[cID_list], [cluster_class], [V, P, D] [normalized V, P, D]
	'''

	if plot_naming == None:
		plot_naming = 'SPEC-debris-network'

	# black, green, blue, magenta, red, yellow, cyan
	color_list = ['rgba(0, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 0, 255, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 1)', 'rgba(0, 255, 255, 1)']
	# color_list = ['rgba(0, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 0, 255, 1)']

	max_resolution = [2000, 5, 0.5, 50, 400]  # volume, velocity, depth, distance, pressure

	DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True)

	## reference data
	# ref_data = [gc.csv2list(csvfileName, starting_row=1) for csvfileName in ref_fileName]

	## reference data
	# ref_data = [gc.csv2list(csvfileName, starting_row=1) for csvfileName in ref_fileName]
	# [time step,X,Y,flowpath travel distance,width,volume,av depth,av vel,max depth,max vel,dynamic P,merged,cluster id,construct]
	# [t(0),X(1),Y(2),s(3),w(4),V(5),av_d(6),av_u(7),max_d(8),max_u(9),P(10),merged(11),cluster id(12),D(13)]

	## 2D contour map
	# plot - source and 3D topo 
	countour_scale = [[0.0, 'rgba(255,255,255,1)'], [0.1, 'rgba(255,255,255,1)'], [0.2, 'rgba(255,255,255,1)'], [0.3, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,1)'], [0.5, 'rgba(255,255,255,1)'], [0.6, 'rgba(255,255,255,1)'], [0.7, 'rgba(255,255,255,1)'], [0.8, 'rgba(255,255,255,1)'], [0.9, 'rgba(255,255,255,1)'], [1.0, 'rgba(255,255,255,1)']]

	topoContour = go.Contour(
		x=gridUniqueX,
		y=gridUniqueY,
		z=DEM,
		line=dict(smoothing=0.85),
		autocontour=False, 
		colorscale=countour_scale,
		showscale = False,
		contours=dict(
			showlabels = True,
			labelfont = dict(
				family = 'Raleway',
				size = 12,
				color = 'black'
			)
			#cmax=np.ceil(max(flowpath.transpose()[loop])),
			#cmin=np.floor(min(flowpath.transpose()[loop]))
		)
	)

	## goal vector
	goal_x = [road_xy_list[0][0], road_xy_list[1][0]]
	goal_y = [road_xy_list[0][1], road_xy_list[1][1]]
	goal_plot = go.Scatter(
		x=goal_x,
		y=goal_y,
		name='goal',
		mode='lines',
		line=dict(
			width=line_width,
			color='rgba(255, 165, 0, 1)'  # orange
		)
	)

	## map-flowpath
	data_path_flowpath = [topoContour, goal_plot]

	## plotting data for map - flowpath
	## SPEC-debris - cluster data
	cID_total = len(flowpath_link.keys())
	for loop_cID in range(cID_total):

		dfl_data = flowpath_link[loop_cID][0]
		dfl_type_data = flowpath_link[loop_cID][1]
		dfl_xy_data = np.array([flowpath_dfl[dfl_id][1] for dfl_id in dfl_data])

		# x,y-coordinates
		path_x = dfl_xy_data[:,0] 
		path_y = dfl_xy_data[:,1] 

		hovertext_flowpath_cluster = [(loop_cID, dfl_i, dfl_type_i) for dfl_i, dfl_type_i in zip(dfl_data, dfl_type_data)]
		cluster_plot_xy = go.Scatter(
			x=path_x,
			y=path_y,
			name='SPEC-debris-cluster'+str(loop_cID), 
			hovertext=hovertext_flowpath_cluster,
			mode='lines+markers',
			line=dict(
				width=line_width,
				color=color_list[loop_cID] 
			),
			marker=dict(
				size=marker_size,
				color=color_list[loop_cID]
			)
		)
		data_path_flowpath.append(cluster_plot_xy)

	
	## optimal barrier location
	for opt_dfl_i in parameter_at_dfl.keys():

		dfl_type_data = flowpath_dfl[opt_dfl_i][0]
		dfl_xy_data = flowpath_dfl[opt_dfl_i][1]
		dfl_V = round(parameter_at_dfl[opt_dfl_i][2][0],dp)
		dfl_P = round(parameter_at_dfl[opt_dfl_i][2][1],dp)
		dfl_D = round(parameter_at_dfl[opt_dfl_i][2][2],dp)

		# x,y-coordinates
		hovertext_opt_barrier = [(opt_dfl_i, dfl_type_data, dfl_V, dfl_P, dfl_D)]
		opt_dfl_plot = go.Scatter(
			x=[dfl_xy_data[0]],
			y=[dfl_xy_data[1]],
			name='optimal_dfl_barrier_'+str(opt_dfl_i), 
			hovertext=hovertext_opt_barrier,
			mode='markers',
			marker=dict(
				size=int(round(marker_size*1.5)),
				color='rgba(255, 0, 0, 1)'
			)
		)
		data_path_flowpath.append(opt_dfl_plot)

	## layout
	layout_map_flowpath = go.Layout(
		title=plot_naming + ' - 2D map - flowpath_network',
		autosize=True,
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=True,
		legend=dict(orientation="h"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		)
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	## figure plotting
	fig_map_flowpath = go.Figure(data=data_path_flowpath, layout=layout_map_flowpath)

	#########################################################################################
	## plot 
	#########################################################################################

	plot(fig_map_flowpath, filename=folder_path+plot_naming+' - 2D map - flowpath_network.html', auto_open=True)

	return None

def plot_flowpath_network_combined_v2_1(folder_path, flowpath_file_name, road_xy_list, flowpath_dfl, flowpath_link, dfl_mitigated_flowpath, plot_naming, dp=2, marker_size=5, line_width=2, layout_width=1000, layout_height=1000):

	'''
	flowpath_link 		# key = cluster id, value = [(0) [linking dfl_ids], (1) [dfl_type]]
	flowpath_dfl  		# key = dfl id, value = [(0) dfl_type, (1) (x,y), (2) [cluster class]]
	dfl_mitigated_flowpath		# key = dfl, value = [[barrier_performance], [cID_list], [V, P, D] [normalized V, P, D]
	'''

	if plot_naming == None:
		plot_naming = 'SPEC-debris-network'

	# black, green, blue, magenta, red, yellow, cyan
	color_list = ['rgba(0, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 0, 255, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 255, 0, 1)', 'rgba(0, 255, 255, 1)']
	# color_list = ['rgba(0, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 0, 255, 1)']

	max_resolution = [2000, 5, 0.5, 50, 400]  # volume, velocity, depth, distance, pressure

	DEM, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True)

	## reference data
	# ref_data = [gc.csv2list(csvfileName, starting_row=1) for csvfileName in ref_fileName]

	## reference data
	# ref_data = [gc.csv2list(csvfileName, starting_row=1) for csvfileName in ref_fileName]
	# [time step,X,Y,flowpath travel distance,width,volume,av depth,av vel,max depth,max vel,dynamic P,merged,cluster id,construct]
	# [t(0),X(1),Y(2),s(3),w(4),V(5),av_d(6),av_u(7),max_d(8),max_u(9),P(10),merged(11),cluster id(12),D(13)]

	## 2D contour map
	# plot - source and 3D topo 
	countour_scale = [[0.0, 'rgba(255,255,255,1)'], [0.1, 'rgba(255,255,255,1)'], [0.2, 'rgba(255,255,255,1)'], [0.3, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,1)'], [0.5, 'rgba(255,255,255,1)'], [0.6, 'rgba(255,255,255,1)'], [0.7, 'rgba(255,255,255,1)'], [0.8, 'rgba(255,255,255,1)'], [0.9, 'rgba(255,255,255,1)'], [1.0, 'rgba(255,255,255,1)']]

	topoContour = go.Contour(
		x=gridUniqueX,
		y=gridUniqueY,
		z=DEM,
		line=dict(smoothing=0.85),
		autocontour=False, 
		colorscale=countour_scale,
		showscale = False,
		contours=dict(
			showlabels = True,
			labelfont = dict(
				family = 'Raleway',
				size = 12,
				color = 'black'
			)
			#cmax=np.ceil(max(flowpath.transpose()[loop])),
			#cmin=np.floor(min(flowpath.transpose()[loop]))
		)
	)

	## goal vector
	goal_x = [road_xy_list[0][0], road_xy_list[1][0]]
	goal_y = [road_xy_list[0][1], road_xy_list[1][1]]
	goal_plot = go.Scatter(
		x=goal_x,
		y=goal_y,
		name='goal',
		mode='lines',
		line=dict(
			width=line_width,
			color='rgba(255, 165, 0, 1)'  # orange
		)
	)

	## map-flowpath
	data_path_flowpath = [topoContour, goal_plot]

	## plotting data for map - flowpath
	## SPEC-debris - cluster data
	cID_total = len(flowpath_link.keys())
	for loop_cID in range(cID_total):

		dfl_data = flowpath_link[loop_cID][0]
		dfl_type_data = flowpath_link[loop_cID][1]
		dfl_xy_data = np.array([flowpath_dfl[dfl_id][1] for dfl_id in dfl_data])

		# x,y-coordinates
		path_x = dfl_xy_data[:,0] 
		path_y = dfl_xy_data[:,1] 

		hovertext_flowpath_cluster = [(loop_cID, dfl_i, dfl_type_i) for dfl_i, dfl_type_i in zip(dfl_data, dfl_type_data)]
		cluster_plot_xy = go.Scatter(
			x=path_x,
			y=path_y,
			name='SPEC-debris-cluster'+str(loop_cID), 
			hovertext=hovertext_flowpath_cluster,
			mode='lines+markers',
			line=dict(
				width=line_width,
				color=color_list[loop_cID] 
			),
			marker=dict(
				size=marker_size,
				color=color_list[loop_cID]
			)
		)
		data_path_flowpath.append(cluster_plot_xy)

	
	## optimal barrier location
	for opt_dfl_i in dfl_mitigated_flowpath.keys():

		dfl_type_data = flowpath_dfl[opt_dfl_i][0]
		dfl_xy_data = flowpath_dfl[opt_dfl_i][1]
		dfl_V = round(dfl_mitigated_flowpath[opt_dfl_i][2][0],dp)
		dfl_P = round(dfl_mitigated_flowpath[opt_dfl_i][2][1],dp)
		dfl_D = round(dfl_mitigated_flowpath[opt_dfl_i][2][2],dp)

		# x,y-coordinates
		hovertext_opt_barrier = [(opt_dfl_i, dfl_type_data, dfl_V, dfl_P, dfl_D)]
		if sum(dfl_mitigated_flowpath[opt_dfl_i][0]) == 2:
			opt_dfl_plot = go.Scatter(
				x=[dfl_xy_data[0]],
				y=[dfl_xy_data[1]],
				name='optimal_dfl_closed_barrier_'+str(opt_dfl_i), 
				hovertext=hovertext_opt_barrier,
				mode='markers',
				marker=dict(
					size=6,
					color='rgba(255, 0, 0, 1)'
				)
			)
		else:
			opt_dfl_plot = go.Scatter(
				x=[dfl_xy_data[0]],
				y=[dfl_xy_data[1]],
				name='optimal_dfl_opened_barrier_'+str(opt_dfl_i)+'_performance_'+str(dfl_mitigated_flowpath[opt_dfl_i][0]), 
				hovertext=hovertext_opt_barrier,
				mode='markers',
				marker=dict(
					size=int(round(marker_size*1.5)),
					color='rgba(128, 0, 128, 1)'
				)
			)
		data_path_flowpath.append(opt_dfl_plot)

	## layout
	layout_map_flowpath = go.Layout(
		title=plot_naming + ' - 2D map - flowpath_network',
		autosize=True,
		width=layout_width, #Nx, #800,
		height=layout_height, #Ny, #1000,
		xaxis=dict(
			# scaleanchor="y",
			# scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='x [m]',
			constrain="domain",
			range=[min(gridUniqueX), max(gridUniqueX)]
		),
		yaxis=dict(
			scaleanchor="x",
			scaleratio=1,
			showline=True,
			linewidth=1, 
			linecolor='black',
			mirror=True,
			autorange=False,
			title='y [m]',
			# constrain="domain",
			range=[min(gridUniqueY), max(gridUniqueY)] 
		),
		showlegend=True,
		legend=dict(orientation="h"),
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		)
		# annotations = dict(
		#     text=plot_annotate,
		#     align='left',
		#     xref='paper',
		#     yref='paper',
		#     x=1.0, 
		#     y=0.1,
		#     showarrow=False
		# )
	)

	## figure plotting
	fig_map_flowpath = go.Figure(data=data_path_flowpath, layout=layout_map_flowpath)

	#########################################################################################
	## plot 
	#########################################################################################

	plot(fig_map_flowpath, filename=folder_path+plot_naming+' 2D map - flowpath_network.html', auto_open=True)

	return None

#################################################################################################################
### main SPEC-debris simulation functions
#################################################################################################################

###########################################################################
## debris-flow simulation
###########################################################################
# for time step approach without barrier performance - cluster and particles
def SPEC_debris_c_t_nb_MP_v9_0(DEM, ERODE, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, cluster_list_i, clusterID_flow_list_i, max_cID_num, all_part_list_i, overall_output_summary_i, road_xy_list, material, COR, Es_theta_var, cell_size, part_radius, l_dp_min, g, merge_overlap_ratio, cluster_boundary, cluster_boundary_iter_dict, interp_method, entrainment_model, t_step, t_max, max_cpu_num, csv_output, exportName, folder_path, DP=4):

	# entrainment_model = 'Hungr' [E(s) = exp(Es*ds)] and 'Er' [E(s) = Es*ds]
	# research from Hansen et al. (2020) - IEEE and energy_conservation_distance of computing energy 
	# where the velocity vector of previous step also influences the flow direction based on weight factor
	# do not compute depth - speed up simulation

	# ref
	# use simulataneous multiple collision between debris-flow particles
	# https://physics.stackexchange.com/questions/296767/multiple-colliding-balls
	# https://en.wikipedia.org/wiki/Coefficient_of_restitution
	# https://en.wikipedia.org/wiki/Collision_detection
	# https://shapely.readthedocs.io/en/latest/manual.html#
	# https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition

	# COR = (COR_p2p, COR_p2w) = (Coefficient of Restitution between particles, Coefficient of Restitution between particle and wall)
	COR_p2p, COR_p2w = COR

	## computation tracking parameters
	# measure of how quickly this solves
	t_track = 0
	time_step = 0

	merged_cID_list = [] 		# store cluster that has merged 

	cluster_ext_polygon = []    # store cluster exterior boundary for plotting later

	start_time = time.time()

	## find decimal place of t_step
	min_dt_dp = abs(decimal.Decimal(str(t_step)).as_tuple().exponent)
 
	## create ghost boundary particles
	ghost_part_list0 = boundary_ghost_part_v1_0(gridUniqueX, gridUniqueY, deltaX, deltaY, part_radius, layers=6)

	try:
		while True: 
			try: 
				###############################################################################
				## all parts
				###############################################################################

				# extract all particles from latest analysis iteration
				cur_all_part_list = all_part_list_i[-1][:]

				# export initial particle data
				if time_step == 0 and csv_output:

					loopT_list = [list(part_i.return_all_param()) for part_i in cur_all_part_list]
					everyT_list = [list(part_i.return_everything()) for part_i in cur_all_part_list]

					if exportName == None:
						exportName = 'SPEC-debris'

					header_every_part = "clusterID,time,si,x,y,z,elevation,r,materialID,phi,fb,ft,rho,Es,dip,dip_direction,travel_direction,cross_travel_direction,gradients_x,gradients_y,dip_travel_direction,cross_dip_travel_direction,Vi,k_MD,k_XMD,ll,hi,divhi,dhdxi,dhdyi,dh_dMD,dh_dXMD,ui,ux,uy,a_MD,a_XMD,sigma,tau_r,dx_grad_local,dy_grad_local,Fr,Pi"
					np.savetxt(folder_path+exportName+'_every_part'+str(0)+'.csv', everyT_list, fmt='%.'+str(DP*2)+'f', delimiter=',', comments='', header=header_every_part)
					del everyT_list

					## parts -  each particle data at each iteration (time_id)
					header_part = 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'
					np.savetxt(folder_path+exportName+'_part'+str(0)+'.csv', loopT_list, fmt='%.'+str(DP*2)+'f', delimiter=',', comments='', header=header_part)

				## multiprocessing set-up
				if mp.cpu_count() >= max_cpu_num:
					cpu_num = max_cpu_num 
				else:
					cpu_num = mp.cpu_count()
				# pool_part = mp.Pool(cpu_num)
	   
				try:
					### potential next location
					pool_part = mp.Pool(cpu_num)

					# from force and pathway
					mp_input_suc = [(part, cell_size, t_step, None, DEM, None, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, g, material, COR_p2w, entrainment_model, Es_theta_var) for part in cur_all_part_list]
					successor_part_list_p1 = pool_part.map(generate_part_successors_mp_v18_0, mp_input_suc)

					pool_part.close()
					pool_part.join()

					# boundary collision check
					boundary_collision_idx = check_boundary_collision_idx(successor_part_list_p1, gridUniqueX, gridUniqueY)
					successor_part_list_p2 = deepcopy(successor_part_list_p1)
					
					if len(boundary_collision_idx) > 0:
						
						pool_part = mp.Pool(cpu_num)
			
						# from boudnary wall collision - continuous collision detection
						# part, cell_size, t_step, wall_dict, DEM, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var, COR_p2w
						boundary_CCD_input = [(successor_part_list_p1[n1], min(cell_size), t_step, None, DEM, None, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var, COR_p2w) for n1 in boundary_collision_idx]
						successor_part_list_p2t = pool_part.map(boundary_CCD_v1_0, boundary_CCD_input)

						pool_part.close()
						pool_part.join()

						for n1, new_p in zip(boundary_collision_idx, successor_part_list_p2t):
							successor_part_list_p2[n1] = deepcopy(new_p)

					### SPH
					ghost_part_list = update_ghost_particle_volume(deepcopy(ghost_part_list0), successor_part_list_p2, max_cpu_num)
					# ghost_part_list = update_ghost_particle_volume(deepcopy(ghost_part_list0), successor_part_list_p2, l_dp_min, max_cpu_num)
		
					# computes: depth(h, dh/dx, dh/dy, eroded depth) and stresses(sigma, tau, mu_eff, diffusion)
					successor_part_sph_list, ERODE = compute_SPH_v4_0(successor_part_list_p2, ghost_part_list, ERODE, material, g, l_dp_min, None, None, gridUniqueX, gridUniqueY, deltaX, deltaY, max_cpu_num)
		
					### collision velocity - for no-barrier situation, assume no p2p collision
					if len(boundary_collision_idx) > 0:
						successor_part_sph_dem_list = part_collision_detection_v5_3(successor_part_sph_list, COR_p2p, cpu_num)
					else:
						successor_part_sph_dem_list = deepcopy(successor_part_sph_list)

					### impact pressure and local gradient
					# multiprocessing set-up
					pool_part = mp.Pool(cpu_num)

					# sort input
					part_p_grad_input = [(part_i, g, t_step) for part_i in successor_part_sph_dem_list]
					# part_p_grad_input = [(part_i, g, t_step) for part_i in successor_part_sph_list]
					successor_part_list = pool_part.map(part_pressure_Fr_grad_local_MP_v1, part_p_grad_input)

					## stop multiprocessing
					pool_part.close()
					pool_part.join()

					# store particle xyz-coordinates
					part_xyz_array = np.array([(part_i.x, part_i.y, part_i.z) for part_i in successor_part_list])

					## export particle data
					if csv_output:

						loopT_list = [list(part_i.return_all_param()) for part_i in successor_part_list]
						everyT_list = [list(part_i.return_everything()) for part_i in successor_part_list]

						if exportName == None:
							exportName = 'SPEC-debris'

						header_every_part = "clusterID,time,si,x,y,z,elevation,r,materialID,phi,fb,ft,rho,Es,dip,dip_direction,travel_direction,cross_travel_direction,gradients_x,gradients_y,dip_travel_direction,cross_dip_travel_direction,Vi,k_MD,k_XMD,ll,hi,divhi,dhdxi,dhdyi,dh_dMD,dh_dXMD,ui,ux,uy,a_MD,a_XMD,sigma,tau_r,dx_grad_local,dy_grad_local,Fr,Pi"
						np.savetxt(folder_path+exportName+'_every_part'+str(time_step+1)+'.csv', everyT_list, fmt='%.8f', delimiter=',', comments='', header=header_every_part)
						del everyT_list

						## parts -  each particle data at each iteration (time_step+1)
						header_part = 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'
						np.savetxt(folder_path+exportName+'_part'+str(time_step+1)+'.csv', loopT_list, fmt='%.4f', delimiter=',', comments='', header=header_part)


				# except:
				except Exception as e:
					## stop multiprocessing
					pool_part.close()
					pool_part.join()

					exc_type, exc_obj, exc_tb = sys.exc_info()
					fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

					print("error cuased by: ")
					print(exc_type, fname, exc_tb.tb_lineno)
					print("terminated early")
		
					print("error cuased by: "+str(e))

					# stop the overall iteration and export simulation
					raise StopIteration()
					break

				###############################################################################
				## all clusters
				###############################################################################
				# create new cluster 
				for cID in clusterID_flow_list_i:
		   
					if cID in merged_cID_list:
						continue

					pre_cluster = cluster_list_i[cID][-1]
					part_in_cur_cluster = [part_i for part_i in successor_part_list if part_i.clusterID == cID]

					# input = clusterID, part_list, predecessor
					next_cluster = Cluster(cID, part_in_cur_cluster, pre_cluster)

					# update new time step
					next_cluster.update_time(d_time=t_step)

					# compute cluster centroid (row,col) and (x,y)
					new_xc, new_yc = next_cluster.compute_centroid()
					# new_xc, new_yc = next_cluster.compute_median_centroid()
					# new_xc, new_yc = next_cluster.compute_max_pressure_centroid()

					# compute elevation - Z
					local_xy_cluster, local_z_cluster = local_cell_v3_0(max(cell_size), new_xc, new_yc, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
					new_zc = compute_Z_v3_0([new_xc, new_yc], local_xy_cluster, local_z_cluster, interp_method)
					next_cluster.zc = new_zc
					# next_cluster.compute_Z(local_xy_cluster, local_z_cluster, interp_method)

					# compute cumnulative cluster volume
					next_cluster.compute_V()

					## compute velocity 
					# particle value average
					next_cluster.compute_av_h_u()	# average depth and velocity

					# compute travel distance (s)
					next_cluster.compute_s()
		
					# compute distance-from-road (D)
					next_cluster.compute_D(road_xy_list)

					# compute pressure
					next_cluster.compute_Fr_P(g)

					## compute external boundary
					# get XY coordinates of the parts contained in the next_cluster
					next_cluster_part_xy_list = [(part.x, part.y) for part in next_cluster.particles]
					
					if cluster_boundary == 'ConcaveHull':	
						# find optimal alpha value and boundary polygon for concave hull using alphashape python library
						cluster_boundary_buffer = part_radius*l_dp_min   # same as the minimum smoothing length

						next_CVH_polygon, next_inlier_pt = determine_optimal_CVH_boundary(next_cluster_part_xy_list, cluster_boundary_buffer=cluster_boundary_buffer, output_inlier_points=True)
						# next_cluster_opt_alpha = alphashape.optimizealpha(next_inlier_pt, max_iterations=cluster_boundary_iter_dict["max iteration"], lower=cluster_boundary_iter_dict["min alpha"], upper=cluster_boundary_iter_dict["max alpha"], silent=True)
						next_cluster_opt_alpha = determine_optimal_alpha(next_inlier_pt, max_alpha=cluster_boundary_iter_dict["max alpha"], min_alpha=cluster_boundary_iter_dict["min alpha"], iter_max=cluster_boundary_iter_dict["max iteration"], dp_accuracy=2)

						next_cluster_boundary_polygon_t = alphashape.alphashape(next_inlier_pt, next_cluster_opt_alpha)
						next_cluster_boundary_polygon = next_cluster_boundary_polygon_t.buffer(cluster_boundary_buffer, cap_style=1, join_style=3)

						next_cluster.concave_hull_alpha = next_cluster_opt_alpha
						next_cluster.boundary_polygon = next_cluster_boundary_polygon

						# boundary XY coordinates and index relative to the input XY coordinate list
						next_cluster_ext_pts_XY = next_cluster_boundary_polygon.exterior.coords.xy
						next_cluster.boundary_pt_xy = next_cluster_ext_pts_XY

					elif cluster_boundary == 'ConvexHull':
						
						# alpha = 0 for Convex hull
						next_cluster.concave_hull_alpha = 0

						# convert corner points to shapely polygon
						cluster_boundary_buffer = part_radius*l_dp_min   # same as the minimum smoothing length
						next_cluster_boundary_polygon = determine_optimal_CVH_boundary(next_cluster_part_xy_list, cluster_boundary_buffer=cluster_boundary_buffer)
						next_cluster.boundary_polygon = next_cluster_boundary_polygon

						# boundary XY coordinates and index relative to the input XY coordinate list
						next_cluster_ext_pts_XY = next_cluster_boundary_polygon.exterior.coords.xy
						next_cluster.boundary_pt_xy = next_cluster_ext_pts_XY

					# compute cluster area
					next_cluster.compute_area()

					# insert into cluster_list
					cluster_list_i[cID].append(next_cluster)

				###############################################
				# Check if a cluster is merging
				###############################################
				if len(clusterID_flow_list_i) > 1:

					merge_list = []
					for cID in clusterID_flow_list_i:
						if cID in merged_cID_list:
							continue 

						next_cID_cluster = cluster_list_i[cID][-1]

						for other_cID in clusterID_flow_list_i[cID:]:
							if other_cID in merged_cID_list:
								continue 

							next_other_cID_cluster = cluster_list_i[other_cID][-1]
							
							merge_bool = check_merge_v5_0(next_cID_cluster, next_other_cID_cluster, merge_overlap_ratio)
							
							if merge_bool:
								merge_list.append((cID, other_cID))
					
					# only one pair merging
					if len(merge_list) > 0:
						
						# check for simulatenous more than 2 clusters merging
						unique_pairing_list = []
						for num, cID_pair in enumerate(merge_list):
							if num == 0:
								unique_pairing_list.append([cID_pair[0], cID_pair[1]])
							else:
								for idx,temp_list in enumerate(unique_pairing_list):
									
									# already same pair exists
									if cID_pair[0] in temp_list and cID_pair[1] in temp_list:
										continue
									
									# one of cluster overlaps
									elif cID_pair[0] in temp_list and cID_pair[1] not in temp_list:
										unique_pairing_list[idx].append(cID_pair[1])
									
									elif cID_pair[0] not in temp_list and cID_pair[1] in temp_list:
										unique_pairing_list[idx].append(cID_pair[0])

									# none of the cluster overlap, merging independent from each other								
									elif cID_pair[0] not in temp_list and cID_pair[1] not in temp_list:
										unique_pairing_list.append(cID_pair)
								
						for cID_pair in unique_pairing_list:
							min_cID = min(cID_pair)
							other_cID = deepcopy(cID_pair)
							other_cID.remove(min_cID)
							other_cID.sort()

							part_in_combined_cluster = [part_i for part_i in successor_part_list if part_i.clusterID in cID_pair]
							part_in_combined_new_cluster_id = [part_ii.replace_clusterID_and_return_class(min_cID) for part_ii in part_in_combined_cluster]

							combined_cluster = Cluster(min_cID, part_in_combined_new_cluster_id, cluster_list_i[min_cID][-2])

							# update new time step
							combined_cluster.update_time(d_time=t_step)

							# compute cluster centroid (row,col) and (x,y)
							combined_xc, combined_yc = combined_cluster.compute_centroid()
							# combined_xc, combined_yc = combined_cluster.compute_median_centroid()
							# combined_xc, combined_yc = combined_cluster.compute_max_pressure_centroid()

							# compute elevation - Z
							local_xy_cluster, local_z_cluster = local_cell_v3_0(max(cell_size), combined_xc, combined_yc, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
							combined_zc = compute_Z_v3_0([combined_xc, combined_yc], local_xy_cluster, local_z_cluster, interp_method)
							combined_cluster.zc = combined_zc

							# compute cumnulative cluster volume
							combined_cluster.compute_V()

							## compute velocity
							# particle value average
							combined_cluster.compute_av_h_u()	# average depth and velocity

							# compute travel distance (s)
							combined_cluster.compute_s()
				
							# compute distance-from-road (D)
							combined_cluster.compute_D(road_xy_list)

							# compute pressure (P)
							combined_cluster.compute_Fr_P(g)

							# notify that is has been merged
							combined_cluster.merged = 1

							## compute external boundary
							# get XY coordinates of the parts contained in the combined_cluster
							combined_cluster_part_xy_list = [(part.x, part.y) for part in combined_cluster.particles]
							
							if cluster_boundary == 'ConcaveHull':	
								# find optimal alpha value and boundary polygon for concave hull using alphashape python library
								cluster_boundary_buffer = part_radius*l_dp_min   # same as smoothing length
								
								combined_CVH_polygon, combined_inlier_pt = determine_optimal_CVH_boundary(combined_cluster_part_xy_list, cluster_boundary_buffer=cluster_boundary_buffer, output_inlier_points=True)
								# combined_cluster_opt_alpha = alphashape.optimizealpha(combined_inlier_pt, max_iterations=cluster_boundary_iter_dict["max iteration"], lower=cluster_boundary_iter_dict["min alpha"], upper=cluster_boundary_iter_dict["max alpha"], silent=True)
								combined_cluster_opt_alpha = determine_optimal_alpha(combined_inlier_pt, max_alpha=cluster_boundary_iter_dict["max alpha"], min_alpha=cluster_boundary_iter_dict["min alpha"], iter_max=cluster_boundary_iter_dict["max iteration"], dp_accuracy=2)

								combined_cluster_boundary_polygon_t = alphashape.alphashape(combined_inlier_pt, combined_cluster_opt_alpha)
								combined_cluster_boundary_polygon = combined_cluster_boundary_polygon_t.buffer(cluster_boundary_buffer, cap_style=1, join_style=3)
																
								combined_cluster.concave_hull_alpha = combined_cluster_opt_alpha
								combined_cluster.boundary_polygon = combined_cluster_boundary_polygon

								# boundary XY coordinates and index relative to the input XY coordinate list
								combined_cluster_ext_pts_XY = combined_cluster_boundary_polygon.exterior.coords.xy
								combined_cluster.boundary_pt_xy = combined_cluster_ext_pts_XY

							elif cluster_boundary == 'ConvexHull':
								
								# alpha = 0 for Convex hull
								combined_cluster.concave_hull_alpha = 0

								# convert corner points to shapely polygon
								cluster_boundary_buffer = part_radius*l_dp_min  # same as smoothing length
								combined_cluster_boundary_polygon = determine_optimal_CVH_boundary(combined_cluster_part_xy_list, cluster_boundary_buffer=cluster_boundary_buffer)
								combined_cluster.boundary_polygon = combined_cluster_boundary_polygon

								# boundary XY coordinates and index relative to the input XY coordinate list
								combined_cluster_ext_pts_XY = combined_cluster_boundary_polygon.exterior.coords.xy
								combined_cluster.boundary_pt_xy = combined_cluster_ext_pts_XY

							# compute cluster area
							combined_cluster.compute_area()

							# insert into cluster_list
							for ccc in cID_pair:
								cluster_list_i[ccc].pop()
								cluster_list_i[ccc].append(combined_cluster)

							# remove from computing cluster
							# always compares larger cluster against cluster with smaller id number
							# therefore, if merging occurs, max_cID is equal to cID (currently analyzing cluster ID)
							print('Merging clusters '+str(cID_pair)+' at time='+str(t_track)+':*************************************************')
							# print('Total time is: %5f' % (time.time() - start_time))  
							
							for ccc_removed in other_cID:
								merged_cID_list.append(ccc_removed)	

				###############################################################################
				## all particles store
				###############################################################################
				# track all part history
				
				# if no merging occurred, simply copy all current particles
				if len(merged_cID_list) == 0:
					all_part_list_i.append(deepcopy(successor_part_list))

				# if merging occurred, current particle clusterID will change
				else:
					successor_part_list_new = []
					for cID in clusterID_flow_list_i:

						if cID in merged_cID_list:
							continue

						cur_cluster = cluster_list_i[cID][-1]
						for part_cID_loop in cur_cluster.particles:
							successor_part_list_new.append(part_cID_loop)

					all_part_list_i.append(deepcopy(successor_part_list_new))

				###############################################################################
				## overall output
				###############################################################################
				# overall centroid
				output_xc, output_yc = np.mean(part_xyz_array[:,[0,1]], axis=0)

				local_xy, local_z = local_cell_v3_0(max(cell_size), output_xc, output_yc, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, None)
				output_zc = compute_Z_v3_0([output_xc, output_yc], local_xy, local_z, interp_method)

				# travel distance (si)
				old_output_xc, old_output_yc, old_output_zc = overall_output_summary_i[-1][2:5]
				output_si = overall_output_summary_i[-1][1] + np.sqrt((output_xc-old_output_xc)**2 + (output_yc-old_output_yc)**2 + (output_zc-old_output_zc)**2)

				# average overall volume, depth and velocity of all particles 
				output_Vi = sum([part_i.Vi for part_i in successor_part_list])
				output_ui = np.mean([part.ui for part in successor_part_list])
				output_hi = np.mean([part.hi for part in successor_part_list])

				# overall output summary
				overall_output_summary_i.append([t_track, output_si, output_xc, output_yc, output_zc, output_ui, output_hi, output_Vi])
						
			except KeyboardInterrupt:
				print()
				print("user terminated early:****************************************************")
				raise StopIteration()
				break
			
			###############################################
			# Check if a cluster is terminating
			###############################################
			# print table head
			if t_track == 0:
				print()
				print('Time [s] \t\t Compute Time [hh:mm:ss] \t\t Expected Time Remaining [hh:mm:ss]') 

			t_track += t_step
			time_step += 1

			# iterations limit
			if t_track >= t_max:
				# continue_process = False
				print()
				print('Reached time max:****************************************************')
				raise StopIteration()
				break

			# compute expected current computed time
			computed_time_s = (time.time() - start_time)
			computed_time_hms = timedelta(seconds=computed_time_s)

			# compute expected time to finish
			av_time_sec_per_t_track = computed_time_s/t_track
			av_remaining_time_s = av_time_sec_per_t_track*max((t_max-t_track),0)
			av_remaining_time_hms = timedelta(seconds=av_remaining_time_s)

			# print current progress
			print(f'{t_track:.2f} \t\t\t {computed_time_hms} \t\t\t\t {av_remaining_time_hms}')
			
	except StopIteration:
		pass
		# raise an exception
		# print('Total simulation time: %2fs' % t_track)
		# print('Total computed time: %5fs' % (time.time() - start_time))

	return cluster_list_i, overall_output_summary_i, all_part_list_i, t_track

# for time step approach with barrier performance - particles
def SPEC_debris_c_t_MP_v9_0(wall_dict, wall_bound_region, DEM_no_wall, DEM_with_wall, ERODE, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, all_part_list_i, material, COR, Es_theta_var, cell_size, part_radius, l_dp_min, g, interp_method, entrainment_model, t_step, t_max, max_cpu_num, csv_output, exportName, folder_path, DP=4):

	# entrainment_model = 'Hungr' [E(s) = exp(Es*ds)] and 'Er' [E(s) = Es*ds]
	# research from Hansen et al. (2020) - IEEE and energy_conservation_distance of computing energy 
	# where the velocity vector of previous step also influences the flow direction based on weight factor
	# do not compute depth - speed up simulation

	# ref
	# use simulataneous multiple collision between debris-flow particles
	# https://physics.stackexchange.com/questions/296767/multiple-colliding-balls
	# https://en.wikipedia.org/wiki/Coefficient_of_restitution
	# https://en.wikipedia.org/wiki/Collision_detection
	# https://shapely.readthedocs.io/en/latest/manual.html#
	# https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition

	# COR = (COR_p2p, COR_p2w) = (Coefficient of Restitution between particles, Coefficient of Restitution between particle and wall)
	COR_p2p, COR_p2w = COR
	# COR_p2p = 0.7  # https://onlinelibrary.wiley.com/doi/epdf/10.1002/nag.2806

	## computation tracking parameters
	# measure of how quickly this solves
	t_track = 0
	time_step = 0

	start_time = time.time()

	## find decimal place of t_step
	min_dt_dp = abs(decimal.Decimal(str(t_step)).as_tuple().exponent)

	## create ghost boundary particles
	ghost_part_list0 = boundary_ghost_part_v1_0(gridUniqueX, gridUniqueY, deltaX, deltaY, part_radius, layers=6)

	try:
		while True: 

			try:
				###############################################################################
				## all parts
				###############################################################################

				# extract all particles from latest analysis iteration
				# cur_all_part_list = all_part_list_i[-1][:]
				cur_all_part_list = deepcopy(all_part_list_i[-1])

				# export initial particle data
				if time_step == 0 and csv_output:

					loopT_list = [list(part_i.return_all_param()) for part_i in cur_all_part_list]
					everyT_list = [list(part_i.return_everything()) for part_i in cur_all_part_list]

					if exportName == None:
						exportName = 'SPEC-debris'

					header_every_part = "clusterID,time,si,x,y,z,elevation,r,materialID,phi,fb,ft,rho,Es,dip,dip_direction,travel_direction,cross_travel_direction,gradients_x,gradients_y,dip_travel_direction,cross_dip_travel_direction,Vi,k_MD,k_XMD,ll,hi,divhi,dhdxi,dhdyi,dh_dMD,dh_dXMD,ui,ux,uy,a_MD,a_XMD,sigma,tau_r,dx_grad_local,dy_grad_local,Fr,Pi"
					np.savetxt(folder_path+exportName+'_every_part'+str(0)+'.csv', everyT_list, fmt='%.'+str(DP*2)+'f', delimiter=',', comments='', header=header_every_part)
					del everyT_list

					## parts -  each particle data at each iteration (time_id)
					header_part = 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'
					np.savetxt(folder_path+exportName+'_part'+str(0)+'.csv', loopT_list, fmt='%.'+str(DP*2)+'f', delimiter=',', comments='', header=header_part)

				## multiprocessing set-up
				if mp.cpu_count() >= max_cpu_num:
					cpu_num = max_cpu_num 
				else:
					cpu_num = mp.cpu_count()
				# pool_part = mp.Pool(cpu_num)

	   
				### potential next location
				pool_part = mp.Pool(cpu_num)

				# from force and pathway
				mp_input_suc = [(part, cell_size, t_step, wall_dict, DEM_no_wall, DEM_with_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, g, material, COR_p2w, entrainment_model, Es_theta_var) for part in cur_all_part_list]
				successor_part_list_p1 = pool_part.map(generate_part_successors_mp_v18_0, mp_input_suc)

				pool_part.close()
				pool_part.join()
	   
				if None in successor_part_list_p1 or isinstance(successor_part_list_p1, list) == False: 
					print(successor_part_list_p1)
					assert False

				## from boudnary wall collision - continuous collision detection
				# boundary collision check
				boundary_collision_idx = check_boundary_collision_idx(successor_part_list_p1, gridUniqueX, gridUniqueY)
				
				successor_part_list_p2 = deepcopy(successor_part_list_p1)

				if len(boundary_collision_idx) > 0:

					pool_part = mp.Pool(cpu_num)
		
					# from boudnary wall collision - continuous collision detection
					# part, cell_size, t_step, wall_dict, DEM_no_wall, DEM_with_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var, COR_p2w
					boundary_CCD_input = [(successor_part_list_p1[n1], min(cell_size), t_step, wall_dict, DEM_no_wall, DEM_with_wall, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, material, entrainment_model, g, Es_theta_var, COR_p2w) for n1 in boundary_collision_idx]
					successor_part_list_p2t = pool_part.map(boundary_CCD_v1_0, boundary_CCD_input)

					pool_part.close()
					pool_part.join()

					for n1, new_p in zip(boundary_collision_idx, successor_part_list_p2t):
						successor_part_list_p2[n1] = deepcopy(new_p)
				
				### SPH 
				ghost_part_list = update_ghost_particle_volume(deepcopy(ghost_part_list0), successor_part_list_p2, max_cpu_num)
				# ghost_part_list = update_ghost_particle_volume(deepcopy(ghost_part_list0), successor_part_list_p2, l_dp_min, max_cpu_num)

				# computes: depth(h, dh/dx, dh/dy, eroded depth) and stresses(sigma, tau, mu_eff, diffusion)
				successor_part_sph_list, ERODE = compute_SPH_v4_0(successor_part_list_p2, ghost_part_list, ERODE, material, g, l_dp_min, wall_dict, wall_bound_region, gridUniqueX, gridUniqueY, deltaX, deltaY, max_cpu_num)

				### collision velocity
				if len(boundary_collision_idx) > 0:
					successor_part_sph_dem_list = part_collision_detection_v5_3(successor_part_sph_list, COR_p2p, cpu_num)
				
				elif len(boundary_collision_idx) == 0 and wall_dict is not None:
					# check particle collision with wall
					part_wall_collision_check = check_all_part_wall_collision(successor_part_sph_list, wall_dict, max_cpu_num)
					if part_wall_collision_check: # at least one particle collided with a wall
						successor_part_sph_dem_list = deepcopy(successor_part_sph_list)
					else:
						successor_part_sph_dem_list = part_collision_detection_v5_3(successor_part_sph_list, COR_p2p, cpu_num)

				elif wall_dict is None and len(boundary_collision_idx) == 0:
					successor_part_sph_dem_list = deepcopy(successor_part_sph_list)

				### impact pressure and local gradient
				# multiprocessing set-up
				pool_part = mp.Pool(cpu_num)

				# sort input
				part_p_grad_input = [(part_i, g, t_step) for part_i in successor_part_sph_dem_list]
				# part_p_grad_input = [(part_i, g, t_step) for part_i in successor_part_sph_list]
				successor_part_list = pool_part.map(part_pressure_Fr_grad_local_MP_v1, part_p_grad_input)

				## stop multiprocessing
				pool_part.close()
				pool_part.join()

				# store part class data
				all_part_list_i.append(deepcopy(successor_part_list))

				## export particle data
				if csv_output:

					loopT_list = [list(part_i.return_all_param()) for part_i in successor_part_list]
					everyT_list = [list(part_i.return_everything()) for part_i in successor_part_list]

					if exportName == None:
						exportName = 'SPEC-debris'

					header_every_part = "clusterID,time,si,x,y,z,elevation,r,materialID,phi,fb,ft,rho,Es,dip,dip_direction,travel_direction,cross_travel_direction,gradients_x,gradients_y,dip_travel_direction,cross_dip_travel_direction,Vi,k_MD,k_XMD,ll,hi,divhi,dhdxi,dhdyi,dh_dMD,dh_dXMD,ui,ux,uy,a_MD,a_XMD,sigma,tau_r,dx_grad_local,dy_grad_local,Fr,Pi"
					np.savetxt(folder_path+exportName+'_every_part'+str(time_step+1)+'.csv', everyT_list, fmt='%.8f', delimiter=',', comments='', header=header_every_part)
					del everyT_list

					## parts -  each particle data at each iteration (time_step+1)
					header_part = 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'
					np.savetxt(folder_path+exportName+'_part'+str(time_step+1)+'.csv', loopT_list, fmt='%.4f', delimiter=',', comments='', header=header_part)

			except KeyboardInterrupt:
				print()
				print('user terminated early:****************************************************')
				raise StopIteration()
				break
			
			###############################################
			# Check if a simulation is terminating
			###############################################
			# print table head
			if t_track == 0:
				print()
				print('Time [s] \t\t Compute Time [hh:mm:ss] \t\t Expected Time Remaining [hh:mm:ss]') 

			t_track += t_step
			time_step += 1

			# iterations limit
			if t_track >= t_max:
				# continue_process = False
				print()
				print('Reached time max:****************************************************')
				raise StopIteration()
				break

			# compute expected current computed time
			computed_time_s = (time.time() - start_time)
			computed_time_hms = timedelta(seconds=computed_time_s)

			# compute expected time to finish
			av_time_sec_per_t_track = computed_time_s/t_track
			av_remaining_time_s = av_time_sec_per_t_track*max((t_max-t_track),0)
			av_remaining_time_hms = timedelta(seconds=av_remaining_time_s)

			# print current progress
			print(f'{t_track:.2f} \t\t\t {computed_time_hms} \t\t\t\t {av_remaining_time_hms}')
			
	except StopIteration:
		pass
		# raise an exception
		# print('Reached time max:****************************************************')
		# print('Total simulation time: %2fs' % t_track)
		# print('Total computed time: %5fs' % (time.time() - start_time))

	return all_part_list_i, t_track

###########################################################################
## debris-flow network - clustering analysis
###########################################################################
def generate_flowpath_network(cluster_list):
	
	'''
	# Note:
	cluster_list = sorted along single flowpath 

	# 5 types of dfl
	0 - dfl
	1 - dfl_start = source
	2 - dfl_merge = merged location (space+time)
	3 - dfl_spatial = spatial merge location (same space, but different time)
	4 - dfl_end = termination location
	'''

	#################################################################################################################
	## store flowpath network data - initial
	#################################################################################################################
	flowpath_link = {}	# key = cluster id, value = [(0) [linking dfl_ids], (1) [dfl_type]]
	flowpath_dfl = {} 	# key = dfl id, value = [(0) dfl_type, (1) (x,y), (2) [cluster class]]
	dfl_number = 0      # store the latest global dfl id number 
	xy = [] 			# idx = dfl_id, value = (x,y),... coordinates in tuple form
	replaced_dfl = {}   # key = previous dfl, value = replaced dfl number

	flowpath_link_new = {}	# key = cluster id, value = [(0) [linking dfl_ids], (1) [dfl_type]]
	flowpath_dfl_new = {} 	# key = dfl id, value = [(0) dfl_type, (1) (x,y), (2) [cluster class]]

	#################################################################################################################
	## find number of dfl types
	#################################################################################################################
	num_dfl_start = len(cluster_list)  # number of sources (i.e. dfl_start) - number of starting locations at time step = 0

	# find the last cluster of each flowpath and check if it merged or terminated (1 = merged, 0 = terminated)
	end_cluster_merge_check = [cluster_list[cID][-1].merged for cID in range(num_dfl_start)]   

	# number of merging
	num_dfl_merge = sum(end_cluster_merge_check)

	# number of termination (i.e. dfl_end) - distance from a urban/road is zero
	num_dfl_end = len(cluster_list) - sum(end_cluster_merge_check)

	# maximum itersation
	num_iters_flowpath = [len(cluster_list[cID]) for cID in range(num_dfl_start)]
	# max_iters = max(num_iters_flowpath)
	# sum_iters = sum(num_iters_flowpath)

	#################################################################################################################
	## sort out link and data for dfl types
	#################################################################################################################
	# iterate at each flowpath and identify as dfl
	# sort only dfl_type = [0, 1, 2, 4]
	for cID in range(num_dfl_start):

		# find minimum distance-from-road for each cluster
		cluster_D_list = [cur_cluster.Dc for cur_cluster in cluster_list[cID]]
		min_cluster_D = min(cluster_D_list)
		min_cluster_D_idx = cluster_D_list.index(min_cluster_D)

		for idx, cur_cluster in enumerate(cluster_list[cID]):

			## assign dfl_type
			if idx == 0:  # source
				cur_cluster.dfl_type = 1

			elif cur_cluster.merged == 1:   # merged dfl
				cur_cluster.dfl_type = 2

			elif idx == min_cluster_D_idx:  # terminal
				cur_cluster.dfl_type = 4

			elif idx > min_cluster_D_idx:  # beyond terminal
				continue  # do not bother with cluster beyond the terminal

			else:  # other type
				cur_cluster.dfl_type = 0

			## dfl type and data into flowpath_link
			# key = cluster id, value = [(0) linking dfl_ids, (1) dfl_type] 
			if idx == 0:
				# dfl id to the flowpath_link dictionary
				flowpath_link[cID] = [[dfl_number], [cur_cluster.dfl_type]]
			else:
				flowpath_link[cID][0].append(dfl_number)
				flowpath_link[cID][1].append(cur_cluster.dfl_type)
			
			## add dfl -> cluster library
			flowpath_dfl[dfl_number] = [cur_cluster.dfl_type, (cur_cluster.xc, cur_cluster.yc), [cur_cluster]]

			## store x,y coordinate of clusters for checking 
			xy.append((cur_cluster.xc, cur_cluster.yc))

			# next dfl id number
			dfl_number += 1

	# print(flowpath_dfl)
	# print(flowpath_link)
	# assert False
	
	## set up nearest neighbor search
	nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(xy)

	## find the other pair dfl type 2 - merge
	if num_dfl_merge > 0: # if merging occured 
		for cID, cID_merge_end in enumerate(end_cluster_merge_check):

			# check if this flowpath terminated with merging
			if cID_merge_end == 1: 

				# find the index, i.e. time step, at which merging occurs
				merged_idx = len(flowpath_link[cID][1])-1   
				
				# find other cluster_flowpath_ID at which merging occurred
				merged_cID_list = [cID]
				for other_cID in range(cID):
					try: 
						if flowpath_link[other_cID][1][merged_idx] == 2:
							merged_cID_list.append(other_cID)
					except:
						continue

				temp_merge_cluster_group = []
				temp_new_xy_group = []
				for flowpath_ID in merged_cID_list:

					# previous dfl_id before change
					old_dfl_number = flowpath_link[flowpath_ID][0][merged_idx]

					# add merged cluster class as new dfl
					for cluster_class in flowpath_dfl[old_dfl_number][2]:
						if len(temp_merge_cluster_group) == 0:
							temp_merge_cluster_group.append(cluster_class)
						
						else:
							cur_cluster_in_group = temp_merge_cluster_group[0]
							if [cluster_class.xc, cluster_class.yc] == [cur_cluster_in_group.xc, cur_cluster_in_group.yc]:
								pass
							else:
								temp_merge_cluster_group.append(cluster_class)
						
					# add merged cluster centroid x-,y-coordinates
					temp_new_xy_group.append(flowpath_dfl[old_dfl_number][1])

					# replace the merged position with new dfl_id
					flowpath_link[flowpath_ID][0][merged_idx] = dfl_number

					# mark which old dfl was replaced to
					replaced_dfl[old_dfl_number] = dfl_number

				## create new dfl group
				# new x and y coordinates
				new_xc = np.mean([x for x,y in temp_new_xy_group])
				new_yc = np.mean([y for x,y in temp_new_xy_group])

				# new flowpath_dfl_database
				flowpath_dfl[dfl_number] = [2, (new_xc, new_yc), temp_merge_cluster_group[:]]
				dfl_number += 1

	# print(flowpath_dfl)
	# print(flowpath_link)
	# print(replaced_dfl)
	# assert False

	## find and group the dfl type 3 - spatial merge 
	# go through each flowpath
	for cID in range(num_dfl_start):

		# other_cID_list = [other_cID for other_cID in range(num_dfl_start) if other_cID != cID]
		# cur_flowpath_dfl_list = flowpath_link[cID][0]

		for idx in range(len(flowpath_link[cID][0])):

			dfl = flowpath_link[cID][0][idx]

			# source or merged region should not have spatial merging - continue
			# if idx == 0 or flowpath_dfl[dfl][0] in [1,2]:
			#	continue

			# source or fully merged region should not have spatial merging - continue
			if idx == 0 or flowpath_dfl[dfl][0] in [1,2]:
				continue

			# other dfl
			elif idx > 0 and flowpath_dfl[dfl][0] in [0,4]:  
				
				## find neighbouring dfl
				cluster_cur_xy = flowpath_dfl[dfl][1] 

				n_neighbors_guess = 5
				while True:
					try:
						distances_dfl, idx_dfl = nbrs.kneighbors([list(cluster_cur_xy)], n_neighbors=n_neighbors_guess)
						break
					except:
						if n_neighbors_guess == 1:
							break
						else:
							n_neighbors_guess -= 1

				distances_dfl = distances_dfl.tolist()
				distances_dfl = distances_dfl[0]
				idx_dfl = idx_dfl.tolist()
				idx_dfl = idx_dfl[0]

				## crit distance
				# we assume that spatially merging dfl should be closer than two dfl before and after current dfl
				prev_dfl_xy = flowpath_dfl[flowpath_link[cID][0][idx-1]][1]
				prev_dist = np.sqrt((prev_dfl_xy[0] - cluster_cur_xy[0])**2 + (prev_dfl_xy[1] - cluster_cur_xy[1])**2)

				# not always we might get a next dfl in the link, so use a general try-except for those cases
				try:
					next_dfl_xy = flowpath_dfl[flowpath_link[cID][0][idx+1]][1]
					next_dist = np.sqrt((next_dfl_xy[0] - cluster_cur_xy[0])**2 + (next_dfl_xy[1] - cluster_cur_xy[1])**2)
				except:
					next_dist = 0
				
				crit_dist = max(prev_dist, next_dist)

				# check if any cur_flowpath_dfl has been replaced 
				replaced_dfl_list = list(replaced_dfl.keys())
				# idx_dfl_2 = [replaced_dfl[idx_i] if idx_i in replaced_dfl_list else idx_i for idx_i in idx_dfl]
				idx_dfl_2 = []
				for idx_i in idx_dfl:
					if idx_i in replaced_dfl_list: 
						if replaced_dfl[idx_i] in replaced_dfl_list:
							if replaced_dfl[replaced_dfl[idx_i]] in replaced_dfl_list:
								idx_dfl_2.append(replaced_dfl[replaced_dfl[replaced_dfl[idx_i]]])
							else:
								idx_dfl_2.append(replaced_dfl[replaced_dfl[idx_i]])
						else:
							idx_dfl_2.append(replaced_dfl[idx_i])
					else:
						idx_dfl_2.append(idx_i)

				## minimum distance 
				# remove dfl from same flowpath - check for only dfl from other flowpath
				# also remove dfl if larger than critical distance of influence
				idx_dfl_other_cID_dist_new = [i for i,idx_i in enumerate(idx_dfl_2) if (idx_i not in flowpath_link[cID][0] and distances_dfl[i] <= crit_dist)]
				if len(idx_dfl_other_cID_dist_new) == 0:
					continue
				else:
					idx_dfl_new = [idx_dfl_2[i] for i in idx_dfl_other_cID_dist_new]
					distances_dfl_new = [distances_dfl[i] for i in idx_dfl_other_cID_dist_new]

				# find the dfl that is closest to the current dfl
				min_dist = min(distances_dfl_new)
				min_dist_dfl = idx_dfl_new[distances_dfl_new.index(min_dist)]
				min_dist_dfl_type = flowpath_dfl[min_dist_dfl][0]

				# if the min dist is non-categorized (dfl_type = 0) or terminus (dfl_type = 4)
				if min_dist_dfl_type in [0, 3, 4]:

					# find the flowpath/cluster that the other dfl is part of 
					temp_cluster_class_list = [flowpath_dfl[min_dist_dfl][2][i] for i in range(len(flowpath_dfl[min_dist_dfl][2])) if flowpath_dfl[min_dist_dfl][2][i].clusterID != cID]
					min_dist_cID_list = [c_class.clusterID for c_class in temp_cluster_class_list] 
									
					if len(min_dist_cID_list) == 0:
						print('some error occurred!!!')
						assert False

					## create a new dfl and replace the old 
					temp_spatial_merge_cluster_group = []
					temp_new_xy_group = []
					for loopN, dfl_n in enumerate([dfl, min_dist_dfl]):

						# add merged cluster class as new dfl
						for cluster_class in flowpath_dfl[dfl_n][2]:
							temp_spatial_merge_cluster_group.append(cluster_class)

						# add merged cluster centroid x-,y-coordinates
						temp_new_xy_group.append(flowpath_dfl[dfl_n][1])

						# replace the merged position with new dfl_id
						if loopN == 0:
							flowpath_link[cID][0][idx] = dfl_number  # replace dfl number

							# if non-category -> tranform into 3
							if min_dist_dfl_type in [0,3]:
								flowpath_link[cID][1][idx] = 3  	     # change dfl_type
							
							# if terminus, remain terminus
							elif min_dist_dfl_type == 4:
								flowpath_link[cID][1][idx] = 4  	     # change dfl_type

						elif loopN == 1:
							for other_cID in min_dist_cID_list:
								
								merged_idx = flowpath_link[other_cID][0].index(dfl_n)
								flowpath_link[other_cID][0][merged_idx] = dfl_number  # replace dfl number
								
								# if non-category -> tranform into 3
								if min_dist_dfl_type in [0,3]:
									flowpath_link[cID][1][idx] = 3  	     # change dfl_type
								
								# if terminus, remain terminus
								elif min_dist_dfl_type == 4:
									flowpath_link[cID][1][idx] = 4  	     # change dfl_type

						# mark which old dfl was replaced to
						replaced_dfl[dfl_n] = dfl_number

					## create new dfl group
					# new x and y coordinates
					new_xc = np.mean([x for x,y in temp_new_xy_group])
					new_yc = np.mean([y for x,y in temp_new_xy_group])

					# new flowpath_dfl_database
					# if non-category -> tranform into 3
					if min_dist_dfl_type in [0,3]:
						flowpath_dfl[dfl_number] = [3, (new_xc, new_yc), temp_spatial_merge_cluster_group[:]]					
					
					# if terminus, remain terminus
					elif min_dist_dfl_type == 4:
						flowpath_dfl[dfl_number] = [4, (new_xc, new_yc), temp_spatial_merge_cluster_group[:]]	
					
					# add new dfl_number
					dfl_number += 1

				# for other dfl_types, continue
				else:
					continue

	#################################################################################################################
	## rearrange flowpath network dfl_id 
	#################################################################################################################
	# reassign all dfl_id so that low dfl number is closer to the flowpath and max dfl_id number = total number of dfl

	# find the maximum time step 
	max_time_step_cID = [len(flowpath_link[cID][0]) for cID in flowpath_link.keys()]
	max_time_step_all = max(max_time_step_cID)

	# generate blank flowpath_link_new = {cID: [[], [dfl_type]]}
	for cID in flowpath_link.keys():
		flowpath_link_new[cID] = [[None for i in range(max_time_step_cID[cID])], [None for i in range(max_time_step_cID[cID])]]

	dfl_number_new = 0
	for t in range(max_time_step_all):

		# compute which cluster to be considered for the specific time step
		loop_cID_list = [cIDi for cIDi in flowpath_link.keys() if t < max_time_step_cID[cIDi]]

		for cID_i in loop_cID_list:

			# if this space was already replaced previously, skip
			if flowpath_link_new[cID_i][0][t] == None:  # don't skip

				# replace from current flowpath
				dfl_number_old = flowpath_link[cID_i][0][t]
				flowpath_link_new[cID_i][0][t] = dfl_number_new
				flowpath_link_new[cID_i][1][t] = flowpath_dfl[dfl_number_old][0]
				flowpath_dfl_new[dfl_number_new] = flowpath_dfl[dfl_number_old][:]	

				# find if dfl_id exists in other flowpath
				find_dfl_number_old_in_other_cID = [cIDii for cIDii in flowpath_link.keys() if dfl_number_old in flowpath_link[cIDii][0]]

				# if in other flowpath, replace them too
				if len(find_dfl_number_old_in_other_cID) > 0:
					for cID_j in find_dfl_number_old_in_other_cID:
						dfl_number_old_cID_idx = flowpath_link[cID_j][0].index(dfl_number_old)
						flowpath_link_new[cID_j][0][dfl_number_old_cID_idx] = dfl_number_new
						flowpath_link_new[cID_j][1][dfl_number_old_cID_idx] = flowpath_dfl[dfl_number_old][0]

				# create a new dfl id number
				dfl_number_new += 1

			else:	# skip as already filled in
				continue

	#################################################################################################################
	## meta-data flowpath network 
	#################################################################################################################

	## network information
	# store dfl number for source and terminus
	source_dfl_list = []
	terminus_dfl_list = []	
	for cID in range(num_dfl_start):
		# source
		source_dfl_list.append(flowpath_link_new[cID][0][0])
		
		# terminus
		if flowpath_link_new[cID][1][-1] == 4 and flowpath_link_new[cID][0][-1] not in terminus_dfl_list:
			terminus_dfl_list.append(flowpath_link_new[cID][0][-1])

	network_data = [num_dfl_start, source_dfl_list, len(terminus_dfl_list), terminus_dfl_list]

	## dfl_candidate
	# list of all dfl unique numbers available as a potential barrier location
	# dfl_candidate = []
	# for cID in range(num_dfl_start):
	# 	for dfls in flowpath_link_new[cID][0]:
	# 		if dfls not in dfl_candidate:
	# 			dfl_candidate.append(dfls)
	# dfl_candidate.sort()  # sort them into numerical order
	dfl_candidate_all = [i for i in range(dfl_number_new)]

	# print(dfl_candidate)
	# print(flowpath_link)
	# print(flowpath_link_new)
	# print(network_data)
	# print(flowpath_dfl_new)
	# assert False

	return flowpath_dfl_new, flowpath_link_new, network_data, dfl_candidate_all


###########################################################################
## optimal closed-type barrier location selection
###########################################################################
## tabu search - Stochastic hill climbing algorithm
def find_optimal_closed_barrier_SHC(flowpath_dfl, flowpath_link, network_data, dfl_candidate, max_VPD, min_VPD, dfl_D, opt_weight, iteration_limit=500, find_iter_limit=200, barrier_num_limit=[None, None], dp=4):

	# import numpy as np
	# from itertools import combinations

	# generate flowpath network
	# flowpath_dfl, flowpath_link, network_data, dfl_candidate = generate_flowpath_network(cluster_list)
	# network_data = [num_dfl_start, source_dfl_list, len(terminus_dfl_list), terminus_dfl_list]

	# max and min debris flow parameters + distance-from-road data
	# max_VPD, min_VPD, dfl_D = VPD_parameters(flowpath_dfl, dfl_candidate, road_xy_list, dp=dp)

	num_dfl_start = network_data[0]
	source_dfl_list = network_data[1] 

	# maximum number of closed barriers to consider
	min_barrier_num = barrier_num_limit[0]
	if min_barrier_num != None:
		if type(min_barrier_num) == int:
			pass
		else:
			TypeError("min_barrier_num should be an integer number higher than 0")
	else:
		min_barrier_num = 1

	max_barrier_num = barrier_num_limit[1]
	if max_barrier_num != None:
		if type(max_barrier_num) == int:
			pass
		else:
			TypeError("max_barrier_num should be an integer number higher than 0")
	else:
		max_barrier_num = num_dfl_start


	# iterate through each number of barriers
	iteration_no = 0
	optimal_dfl_barrier = {}   # store optimal dfl for each b_no
	step_recording = {}

	for b_no in range(min_barrier_num, max_barrier_num+1):

		# max_iterable = min([iteration_limit, len(list(combinations(dfl_candidate, b_no)))])
		max_iterable = min([iteration_limit, comb(len(dfl_candidate), b_no, exact=True)])

		checked_dfl_set = []
		passed_dfl_set = []
		checked_dfl_set_cost = []
		old_dfl_barrier = []

		best_dfl_barrier = []
		track_best_dfl_barrier = []

		# repeat_num = 0
		for iteration_no in range(max_iterable):

			# if repeat_num > repeat_limit:
			# 	break

			# for 1st time generate random placement of barriers
			# for next time use hill climbing algorithm to exchange one dfl into new one and check if there is improvement
			# if not improvement, then keep the current. if improvement shown, keep the new one
			dfl_barrier_set = assign_dfl_barrier_sets(iteration_no, b_no, old_dfl_barrier, dfl_candidate, flowpath_link, num_dfl_start, checked_dfl_set, find_iter_limit)
			checked_dfl_set.append(dfl_barrier_set[:])

			# check whether the barrier has successfully blocked
			# dfl_barrier_set = [18, 27, 43]
			goal_achieved, cost, dfl_mitigated_flowpath = cost_closed_barrier_location(dfl_barrier_set, flowpath_link, flowpath_dfl, max_VPD, min_VPD, dfl_D, opt_weight, dp=dp)
			
			# if goal_achieved, then perform cost
			if goal_achieved: # mitigation successful

				passed_dfl_set.append(dfl_barrier_set[:])
				checked_dfl_set_cost.append(cost)

				# if there is only one possible dfl, then it is replaced each time, then later the optimal one is found
				if b_no == 1: 
					continue

				elif iteration_no == 0:
					old_dfl_barrier = dfl_barrier_set[:]
					best_dfl_barrier = [dfl_barrier_set[:], cost]

				else:
					# improvement, replace the current best dfl_barrier set
					if cost < best_dfl_barrier[1]:
						old_dfl_barrier = dfl_barrier_set[:]
						best_dfl_barrier = [dfl_barrier_set[:], cost]
					
					# no improvement, then keep the current best dfl_barrier set
					else:
						old_dfl_barrier = best_dfl_barrier[0][:]

			else: # mitigation unsuccessful
				if iteration_no == 0:
					old_dfl_barrier = dfl_barrier_set[:]
					best_dfl_barrier = [dfl_barrier_set[:], 100]
				else:
					old_dfl_barrier = best_dfl_barrier[0][:]
				
				checked_dfl_set_cost.append(100)
				passed_dfl_set.append(None)

			track_best_dfl_barrier.append(best_dfl_barrier)

		# print(checked_dfl_set_cost, passed_dfl_set)

		# if there is only one possible dfl - simply iterative search/brute force
		if b_no == 1:
			min_cost = min(checked_dfl_set_cost)
			min_cost_idx = checked_dfl_set_cost.index(min_cost)

			if min_cost == 100:
				optimal_dfl_barrier_b1 = None
				optimal_dfl_barrier[b_no] = [None, 100]
			else:
				optimal_dfl_barrier_b1 = passed_dfl_set[min_cost_idx]
				optimal_dfl_barrier[b_no] = [optimal_dfl_barrier_b1[:], min_cost]

		else:
			optimal_dfl_barrier[b_no] = best_dfl_barrier[:]

		## record each step for later analysis - (e.g.) benchmark
		temp_record_p = []
		for check, cost in zip(checked_dfl_set, checked_dfl_set_cost):
			temp_record = []
			temp_record.append(cost)
			for dfl in check:
				temp_record.append(dfl)
			temp_record_p.append(temp_record)

		temp_best_p = []
		for best in track_best_dfl_barrier:
			temp_best = []
			temp_best.append(best[1])
			for dfl in best[0]:
				temp_best.append(dfl)
			temp_best_p.append(temp_best)

		step_recording[b_no] = [temp_record_p[:], temp_best_p[:]]

	## find the optimal - dfl_set, cost
	most_optimal_dfl_barrier = []

	for b_no in range(min_barrier_num, max_barrier_num+1):

		if optimal_dfl_barrier[b_no][0] == None:
			continue

		else:
			if len(most_optimal_dfl_barrier) == 0:
				most_optimal_dfl_barrier = optimal_dfl_barrier[b_no][:]

			else:
				if most_optimal_dfl_barrier[1] > optimal_dfl_barrier[b_no][1]:
					most_optimal_dfl_barrier = optimal_dfl_barrier[b_no][:]

	## return optimal barrier locations and their cost
	# return most_optimal_dfl_barrier[0], most_optimal_dfl_barrier[1], best_min_cost_dfl_mitigated_flowpath, optimal_dfl_barrier, step_recording
	return most_optimal_dfl_barrier[0], most_optimal_dfl_barrier[1], optimal_dfl_barrier, step_recording

## iterate a few time to avoid local minimum
def iterate_find_optimal_closed_barrier_v2_0(folder_path, flowpath_file_name, cluster_list, road_xy_list, total_loop=10, opt_weight=[0.35, 0.2, 0.15, 0.3], avoid_dfl=True, iteration_limit=500, find_iter_limit=200, barrier_num_limit=[None, None], output_step=False, export_step=False, plot_network=False, exportName=None, marker_size=5, line_width=2, layout_width=1000, layout_height=1000, dp=4):

	# import numpy as np 
	# import time

	## generate network
	flowpath_dfl, flowpath_link, network_data, dfl_candidate_all = generate_flowpath_network(cluster_list)
	max_VPD, min_VPD, dfl_D = VPD_parameters(flowpath_dfl, dfl_candidate_all, road_xy_list, dp=dp)
	# print(flowpath_link)

	## update  
	if avoid_dfl:  
		dfl_candidate = generate_avoid_dfl_closed(flowpath_link)
	else:  # avoid only source dfl
		dfl_candidate = [dfl_i for dfl_i in dfl_candidate_all if dfl_i not in network_data[1]]
	# print(dfl_candidate)

	## final optimal dfl closed-type barrier locations
	definite_optimal_dfl_barrier = []

	# o_s_t = time.time()

	# i_s_t_list = []
	if output_step:
		output_dict_list = []

	for i in range(total_loop):
		# print(i)
		# i_s_t = time.time()

		# stochastic hill climbing algorithm
		opt_dfl_locations, opt_dfl_cost, optimal_dfl_barrier, step_recording = find_optimal_closed_barrier_SHC(flowpath_dfl, flowpath_link, network_data, dfl_candidate, max_VPD, min_VPD, dfl_D, opt_weight, iteration_limit=iteration_limit, barrier_num_limit=barrier_num_limit, dp=dp)

		# print(i, opt_dfl_locations, opt_dfl_cost)

		if output_step:
			output_dict_list.append([optimal_dfl_barrier, step_recording])

		if len(definite_optimal_dfl_barrier) == 0:
			definite_optimal_dfl_barrier = [opt_dfl_locations, opt_dfl_cost]

		else:
			if definite_optimal_dfl_barrier[1] > opt_dfl_cost:
				definite_optimal_dfl_barrier = [opt_dfl_locations, opt_dfl_cost]

			elif definite_optimal_dfl_barrier[1] == opt_dfl_cost:
				choice_list = [definite_optimal_dfl_barrier[:], [opt_dfl_locations, opt_dfl_cost]]
				
				if len(opt_dfl_locations) == len(definite_optimal_dfl_barrier[0]):
					choice_index = np.random.randint(0, 2, 1, dtype=int)
					definite_optimal_dfl_barrier = choice_list[choice_index[0]][:]

				elif len(opt_dfl_locations) < len(definite_optimal_dfl_barrier[0]):
					definite_optimal_dfl_barrier = [opt_dfl_locations, opt_dfl_cost]

		# i_e_t = time.time()
		# i_s_t_list.append(round(i_e_t - i_s_t, 4))

		# print(opt_dfl_locations, opt_dfl_cost, definite_optimal_dfl_barrier) 

		if export_step:
			for b_no in optimal_dfl_barrier.keys():
				export_format = '%.'+str(dp)+'f'
				export_name_opt = 'opt_loop_'+str(i)+'_total_loop_'+str(total_loop)+'-limit_'+str(iteration_limit)+'-b_no_'+str(b_no)+'-SHC-opt_weight_'+str(opt_weight)+'.txt'
				export_comments = 'b_no = '+str(b_no)+' optimal DFL = '+str(optimal_dfl_barrier[b_no][0])+' cost = '+str(optimal_dfl_barrier[b_no][1])

				if b_no == 1:
					np.savetxt(folder_path+export_name_opt, np.array(step_recording[b_no][0]), fmt=export_format, delimiter='\t', comments='', header=export_comments)
				elif b_no > 1:
					np.savetxt(folder_path+export_name_opt, np.array(step_recording[b_no][1]), fmt=export_format, delimiter='\t', comments='', header=export_comments)

	## debris-flow parameters
	## compute the most_optimal_dfl_barrier to get parameters
	goal_achieved, best_min_cost, parameter_at_dfl = cost_closed_barrier_location(definite_optimal_dfl_barrier[0][:], flowpath_link, flowpath_dfl, max_VPD, min_VPD, dfl_D, opt_weight, dp=dp)
	
	# check the final parameters_at_dfl - remove redundant design
	final_opt_dfl = {}
	final_optimal_dfl_barrier = []
	for dfl_o in parameter_at_dfl.keys():
		if sum(parameter_at_dfl[dfl_o][2]) > 0:
			final_opt_dfl[dfl_o] = parameter_at_dfl[dfl_o][:]
			final_optimal_dfl_barrier.append(dfl_o)

	## plot optimal barrier location and network flowpath
	if plot_network:
		plot_flowpath_network_closed_v2_1(folder_path, flowpath_file_name, road_xy_list, flowpath_dfl, flowpath_link, final_opt_dfl, exportName, dp=dp, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

	return final_optimal_dfl_barrier, best_min_cost, final_opt_dfl


###########################################################################
## optimal combined type barrier location selection
###########################################################################
## Stochastic hill climbing algorithm - multiprocessing
def find_optimal_combined_barrier_SHC_MP_child(find_optimal_combined_barrier_SHC_MP_child_input):

	# sort input
	b_no, flowpath_dfl, flowpath_link, network_data, dfl_candidate, max_VPD, min_VPD, dfl_D, opt_weight, open_performance, cell_size, material, MAT, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, Es_theta_var, fb_0, g, alpha, entrainment_model, interp_method, iteration_limit, find_iter_limit, min_uV, VI_crit, RC_bool, dp, output_step = find_optimal_combined_barrier_SHC_MP_child_input

	# b_no, flowpath_dfl, flowpath_link, network_data, dfl_candidate, max_VPD, min_VPD, dfl_D, opt_weight, open_performance, cell_size, DEM, gridUniqueX, gridUniqueY, f, Es, alpha, density, g, entrainment_model, interp_method, iteration_limit, find_iter_limit, min_uV, VI_crit, RC_bool, dp = find_optimal_combined_barrier_SHC_MP_child_input

	# import numpy as np
	# from itertools import combinations

	# generate flowpath network
	# flowpath_dfl, flowpath_link, network_data, dfl_candidate = generate_flowpath_network(cluster_list)
	# network_data = [num_dfl_start, source_dfl_list, len(terminus_dfl_list), terminus_dfl_list]

	# max and min debris flow parameters + distance-from-road data
	# max_VPD, min_VPD, dfl_D = VPD_parameters(flowpath_dfl, dfl_candidate, road_xy_list, dp=dp)

	num_dfl_start = network_data[0]

	# iterate through each number of barriers	
	max_iterable = min([iteration_limit, comb(len(dfl_candidate), b_no, exact=True)])

	checked_open_dfl_set = []
	old_dfl_barrier = []		# [[open], [closed]]

	if output_step:
		output_dict_list = []

	# repeat_num = 0
	for iteration_no in range(max_iterable):

		# for 1st time generate random placement of barriers
		# for next time use hill climbing algorithm to exchange one dfl into new one and check if there is improvement
		# if not improvement, then keep the current. if improvement shown, keep the new one

		dfl_barrier_set, dfl_uhVP_dict, cID_uhVP_dict, dfl_terminus_VI = assign_dfl_combined_barrier_sets_v3_0(iteration_no, b_no, old_dfl_barrier, num_dfl_start, checked_open_dfl_set, find_iter_limit, dfl_candidate, flowpath_link, flowpath_dfl, network_data, open_performance, cell_size, g, material, MAT, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, entrainment_model, interp_method, min_uV, Es_theta_var, fb_0, VI_crit, RC_bool)
		checked_open_dfl_set.append(deepcopy(dfl_barrier_set[0]))
		
		# check whether the barrier has successfully blocked
		# alpha = 1.5
		cost, dfl_mitigated_flowpath = cost_combined_barrier_location_v1_0(dfl_barrier_set, flowpath_dfl, cID_uhVP_dict, dfl_uhVP_dict, open_performance, max_VPD, min_VPD, dfl_D, alpha, material, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, opt_weight, dp=dp)

		# elif iteration_no == 0:
		if iteration_no == 0:
			old_dfl_barrier = deepcopy(dfl_barrier_set)
			best_dfl_barrier = [deepcopy(dfl_barrier_set), cost]
			best_dfl_barrier_data = [deepcopy(dfl_mitigated_flowpath), deepcopy(dfl_terminus_VI)]

		else:
			# improvement, replace the current best dfl_barrier set
			if cost < best_dfl_barrier[1]:
				old_dfl_barrier = deepcopy(dfl_barrier_set)
				best_dfl_barrier = [deepcopy(dfl_barrier_set), cost]
				best_dfl_barrier_data = [deepcopy(dfl_mitigated_flowpath), deepcopy(dfl_terminus_VI)]
			
			# no improvement, then keep the current best dfl_barrier set
			else:
				old_dfl_barrier = deepcopy(best_dfl_barrier[0])
		
		if output_step:
			output_str_list = [str(best_dfl_barrier[0][0]), str(best_dfl_barrier[0][1]), best_dfl_barrier[1]]
			output_dict_list.append(output_str_list)
			# output_dict_list.append([deepcopy(dfl_barrier_set[0]), deepcopy(dfl_barrier_set[1]), deepcopy(cost)])

		del dfl_barrier_set
		del dfl_uhVP_dict
		del cID_uhVP_dict
		del dfl_terminus_VI
		del cost
		del dfl_mitigated_flowpath

	## return optimal barrier locations, cost, and other relevent data
	# output = [optimal dfl set, cost, dfl_mitigated_flowpath, dfl_terminus_VI]
	
	if output_step:
		output = (deepcopy(best_dfl_barrier), deepcopy(best_dfl_barrier_data), output_dict_list)
	else:
		output = (deepcopy(best_dfl_barrier), deepcopy(best_dfl_barrier_data))

	return output

## iterate a few time to avoid local minimum - multiprocessing
def iterate_find_optimal_combined_barrier_v3_0(folder_path, flowpath_file_name, cluster_list, road_xy_list, material, MAT, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, fb_0, alpha=1.5, total_loop=10, open_performance=[0.4, 0.4], opt_weight=[0.35, 0.2, 0.15, 0.1, 0.2], avoid_dfl=True, iteration_limit=500, find_iter_limit=200, cell_size=5, g=9.81, entrainment_model='Hungr', Es_theta_var=5, interp_method='linear', min_uV=[0,0], VI_crit=0.7, RC_bool=True, barrier_num_limit=[None, None], output_step=False, plot_network=False, exportName=None, max_cpu_num=16, marker_size=5, line_width=2, layout_width=1000, layout_height=1000, dp=4):

	# import numpy as np 
	# import time

	## generate network
	flowpath_dfl, flowpath_link, network_data, dfl_candidate_all = generate_flowpath_network(cluster_list)
	max_VPD, min_VPD, dfl_D = VPD_parameters(flowpath_dfl, dfl_candidate_all, road_xy_list, dp=dp)

	## update  
	if avoid_dfl:  
		dfl_candidate = generate_avoid_dfl_combined(flowpath_link)
	else:  # avoid source and terminus dfl for open-type barriers
		dfl_candidate = [dfl_i for dfl_i in dfl_candidate_all if (dfl_i not in network_data[1]) and (dfl_i not in network_data[3])]
	dfl_candidate.sort()
	# print(dfl_candidate)

	## final optimal dfl closed-type barrier locations
	definite_optimal_dfl_barrier = []
	definite_optimal_dfl_barrier_data = []

	# maximum number of closed barriers to consider
	min_barrier_num = barrier_num_limit[0]
	if min_barrier_num != None:
		if type(min_barrier_num) == int:
			pass
		else:
			TypeError("min_barrier_num should be an integer number higher than 0")
	else:
		min_barrier_num = int(network_data[2])

	max_barrier_num = barrier_num_limit[1]
	if max_barrier_num != None:
		if type(max_barrier_num) == int:
			pass
		else:
			TypeError("max_barrier_num should be an integer number higher than 0")
	else:
		max_barrier_num = int(np.ceil(network_data[0]*1.5))

	## grid X and Y
	if isinstance(gridUniqueX, list):
		gridUniqueX = np.array(gridUniqueX)

	if isinstance(gridUniqueY, list):
		gridUniqueY = np.array(gridUniqueY)

	## start multiprocessing
	cpu_num = min(mp.cpu_count(), max_cpu_num)
	pool_SHC_loop = mp.Pool(cpu_num)

	## iterate through each number of barriers			
	# create input
	find_optimal_combined_barrier_SHC_MP_child_input = []
	for loop_no in range(total_loop):
		for b_no in range(min_barrier_num, max_barrier_num+1):
			find_optimal_combined_barrier_SHC_MP_child_input.append((b_no, flowpath_dfl, flowpath_link, network_data, dfl_candidate, max_VPD, min_VPD, dfl_D, opt_weight, open_performance, cell_size, material, MAT, DEM, gridUniqueX, gridUniqueY, deltaX, deltaY, Es_theta_var, fb_0, g, alpha, entrainment_model, interp_method, iteration_limit, find_iter_limit, min_uV, VI_crit, RC_bool, dp, output_step))

	optimal_dfl_barrier_results = pool_SHC_loop.map(find_optimal_combined_barrier_SHC_MP_child, find_optimal_combined_barrier_SHC_MP_child_input)

	## stop multiprocessing
	pool_SHC_loop.close()
	pool_SHC_loop.join()

	## select the choice with minimum cost
	# list of cost for each loop
	total_loop_cost_list = []
	for nn in range(len(optimal_dfl_barrier_results)):
		total_loop_cost_list.append(optimal_dfl_barrier_results[nn][0][1])
	min_total_loop_cost = min(total_loop_cost_list)
	min_total_loop_cost_idx = total_loop_cost_list.index(min_total_loop_cost)

	definite_optimal_dfl_barrier = deepcopy(optimal_dfl_barrier_results[min_total_loop_cost_idx][0])
	definite_optimal_dfl_barrier_data = deepcopy(optimal_dfl_barrier_results[min_total_loop_cost_idx][1])

	# o_e_t = time.time()
	## debris-flow parameters	

	## plot optimal barrier location and network flowpath
	if plot_network:
		plot_flowpath_network_combined_v2_1(folder_path, flowpath_file_name, road_xy_list, flowpath_dfl, flowpath_link, definite_optimal_dfl_barrier_data[0], exportName, dp=dp, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

	## export cost per iteration 
	if output_step:
		for nn in range(len(optimal_dfl_barrier_results)):
			b_no = find_optimal_combined_barrier_SHC_MP_child_input[nn][0]
			# export_format = '%.'+str(dp)+'f'
			export_name_opt = 'loop_'+str(nn)+'-limit_-b_no_'+str(b_no)+'-SHC-opt_weight_'+str(opt_weight)+'.txt'
			export_comments = 'b_no = '+str(b_no)+' optimal DFL = '+str(definite_optimal_dfl_barrier[0])+' cost = '+str(definite_optimal_dfl_barrier[1])
			# np.savetxt(folder_path+export_name_opt, np.array(optimal_dfl_barrier_results[nn][2]), fmt=export_format, delimiter='\t', comments='', header=export_comments)

			with open(folder_path+export_name_opt, 'w', newline='') as csvfile:
				writer = csv.writer(csvfile, delimiter='\t')
				writer.writerow([export_comments])  # comment
				writer.writerow(['opened DFL', 'closed DFL', 'cost'])  # header
				for data in optimal_dfl_barrier_results[nn][2]:
					writer.writerow(data)   # track the optimal barrier location sets for each iteration
			csvfile.close()

	## return - optimal_dfl_combined_barrier_set, combined_barrier_cost, combined_barrier_dfl_data, terminus_dfl_data
	return definite_optimal_dfl_barrier[0], definite_optimal_dfl_barrier[1], definite_optimal_dfl_barrier_data[0], definite_optimal_dfl_barrier_data[1]

###########################################################################
## check JSON input files - check json input file to be valid or not
###########################################################################
# SPEC-debris-closed 
def check_SPEC_debris_closed_json_input_v11_00(json_input_number, json_file_name, json_input_data):

	##################################################################################################################
	## read json file and extract input file data - also check whether JSON input file has no issues
	##################################################################################################################

	try: # exception for json input file error

		###########################################################
		## Name and folder information
		###########################################################
		# project name / export name
		exportName = json_input_data["project name"] 

		# project folder path
		folder_path = json_input_data["folder path"]
		if folder_path == None:  # current multiple json file folder path == each simulation folder path
			folder_path = os.path.dirname(json_file_name)+'/'
		else:
			folder_path = folder_path+'/'

		###########################################################
		## map data - DEM, road, wall/building
		###########################################################
		# DEM data
		flowpath_file_name = json_input_data["flowpath file name"]
		source_file_name = json_input_data["source file name"]
		material_file_name = json_input_data["material file name"] 

		# road and goal XY-coordinates
		road_xy_list = [tuple(xy_pt) for xy_pt in json_input_data["road xy"]]

		###########################################################
		## SPEC-debris analysis set-up and material properties
		###########################################################
		# time step parameters
		t_step = json_input_data["time step interval"]
		t_max = json_input_data["maximum simulation time"]
		
		# particle number
		if json_input_data["particle number per cell"] is None:
			part_num_per_cell = 1
		else:
			try:
				part_num_per_cell = int(json_input_data["particle number per cell"])
			except:
				part_num_per_cell = 1

		# initial setup of debris-flow - initial velocity
		initial_velocity = (json_input_data["initial u_x"], json_input_data["initial u_y"])

		# material
		material = {}
		for mat_key, mat_info in json_input_data["material"].items():
			material[int(mat_key)] = mat_info

		# load - gravity
		g = json_input_data["gravitational acceleration"]

		# interaction - coefficient of restitution COR
		COR = (json_input_data["coefficient of restitution (COR)"]["particle with particle"], 
				json_input_data["coefficient of restitution (COR)"]["particle with wall"])
		# COR = (1.0, json_input_data["coefficient of restitution (COR)"]["particle with wall"])

		# perform SPH interpolation at time step = 0
		initial_SPH = json_input_data["initial SPH"] 

		###########################################################
		## Algorithm options 
		###########################################################
		# interpolation method
		interp_method = json_input_data["interpolation method"]

		# Entrainment model
		entrainment_model = json_input_data["entrainment model"]

		# cluster boundary algorithm
		cluster_boundary = json_input_data["cluster boundary method"]	

		###########################################################
		## Control parameters 
		###########################################################
		# no entrainment when climbing wall or free-falling
		Es_theta_var = json_input_data["free fall angle variation"]

		# simulation coefficients and constants
		cell_size = tuple(json_input_data["local cell sizes"]) 		

		# SPH smoothing length computation
		l_dp_min = json_input_data["l_dp_min"]

		# cluster boundary control parameters for Concave-Hull
		concave_hull_parameter_dict = json_input_data["concave hull algorithm parameter"]

		# mutliple cluster merge
		merge_overlap_ratio = json_input_data["merge overlap ratio"]

		# multiprocessing
		max_cpu_num = json_input_data["max cpu num"]

		###########################################################
		## optimal barrier location options - closed-type only
		###########################################################

		# selection of optimal barrier parameters 
		opt_total_loop = json_input_data["optimal barrier location selection option"]["optimal total loop"]
		opt_avoid_dfl = json_input_data["optimal barrier location selection option"]["optimal avoid dfl"]
		opt_iter_max = json_input_data["optimal barrier location selection option"]["optimal iteration max"]
		opt_find_iter_limit = json_input_data["optimal barrier location selection option"]["optimal find iteration limit"]
		opt_barrier_num_limit = [json_input_data["optimal barrier location selection option"]["optimal barrier num min"],
									json_input_data["optimal barrier location selection option"]["optimal barrier num max"]]

		# track cost per epoch
		output_optimal_step = json_input_data["output optimal step"]

		# only closed-type barrier
		opt_weight_close = [
			json_input_data["closed barrier optimization criteria"]["volume(V)"],
			json_input_data["closed barrier optimization criteria"]["pressure(P)"],
			json_input_data["closed barrier optimization criteria"]["distance_from_road(D)"],
			json_input_data["closed barrier optimization criteria"]["closed_barrier_number(N)"]
		]
		
		###########################################################
		## plotting options 
		###########################################################
		
		# plotting options
		plot_map_2D = json_input_data["static plot option"]["plot map 2D"]
		plot_map_3D = json_input_data["static plot option"]["plot map 3D"]
		
		# animation options
		plot_animation_2D = json_input_data["animation option"]["plot animation 2D"]
		plot_animation_3D = json_input_data["animation option"]["plot animation 3D"]
		plot_animation_2D_boundary = json_input_data["animation option"]["plot animation 2D boundary"]

		# open plot after creating
		open_plot = json_input_data["open plot"]

		# 3D plot option
		plot_3D_z_offset = json_input_data["static plot option"]["plot 3D z offset"]

		# animation options
		animation = [
			json_input_data["animation option"]["frame duration"], 
			json_input_data["animation option"]["frame transition"], 
			json_input_data["animation option"]["contour elevation interval"]
		]
		animation_3D_z_offset = json_input_data["animation option"]["animation 3D z offset"]

		# plot maximum legend
		plot_2D_max_limits = [
			json_input_data["max parameter legend"]["cluster max velocity(u)"],
			json_input_data["max parameter legend"]["cluster max depth(h)"],
			json_input_data["max parameter legend"]["cluster max volume(V)"],
			json_input_data["max parameter legend"]["cluster max distance_from_road(D)"],
			json_input_data["max parameter legend"]["cluster max pressure(P)"],
			json_input_data["max parameter legend"]["particle max velocity(u)"],
			json_input_data["max parameter legend"]["particle max depth(h)"],
			json_input_data["max parameter legend"]["particle max volume(V)"],
			json_input_data["max parameter legend"]["particle max pressure(P)"]
		]
		plot_3D_max_limits = deepcopy(plot_2D_max_limits)

		# marker, line, and plot size options
		marker_size = json_input_data["marker size"]
		line_width = json_input_data["line width"]
		layout_width = json_input_data["layout width"]
		layout_height = json_input_data["layout height"]

		###########################################################
		## output options 
		###########################################################
		# csv data export for SPEC-debris analysis
		csv_output = json_input_data["csv output"]

		# data decimal places
		dp = json_input_data["decimal points"]

	except (KeyError, NameError):
		return json_input_number*1000+2	# input file variable error

	##################################################################################################################
	## check valid input file type, existance, and value
	##################################################################################################################

	##########################################################
	## flowpath_file_name
	##########################################################
	# check if it exists or has valid naming
	if os.path.isfile(folder_path+flowpath_file_name) == False:
		return json_input_number*1000+20	# flowpath file error

	# check file type
	flowpath_file_name_list = flowpath_file_name.split('.')
	flowpath_file_name_type = flowpath_file_name_list[-1]
	if flowpath_file_name_type not in ['csv', 'las', 'grd']:
		return json_input_number*1000+21	# flowpath filetype error

	##########################################################
	## source_file_name
	##########################################################
	# check if it exists or has valid naming
	if os.path.isfile(folder_path+source_file_name) == False:
		return json_input_number*1000+30	# source file error

	# check file type
	source_file_name_list = source_file_name.split('.')
	source_file_name_type = source_file_name_list[-1]
	if source_file_name_type not in ['csv', 'las', 'grd']:
		return json_input_number*1000+31	# source filetype error

	##############################################################
	## material_file_name
	##############################################################
	# import material file if number of number of material is higher than 1
	if material_file_name is not None and len(material.keys()) > 1:

		# check if it exists or has valid naming
		if os.path.isfile(folder_path+material_file_name) == False:
			return json_input_number*1000+40	# material file error

		# check file type
		material_file_name_list = material_file_name.split('.')
		material_file_name_type = material_file_name_list[-1]
		if material_file_name_type not in ['csv', 'las', 'grd']:
			return json_input_number*1000+41	# material filetype error

	# missing material data 
	elif material_file_name is None and len(material.keys()) > 1:
		return json_input_number*1000+42	# material file required

	##############################################################
	## road
	##############################################################
	# road format test
	try: 
		if not isinstance(road_xy_list, list):
			return json_input_number*1000+50	# road xy points error
	except:
		return json_input_number*1000+50		# road xy points error

	road_isnumeric_check = int(sum([isinstance(rxy[0], (int,float)) + isinstance(rxy[1], (int,float)) for rxy in road_xy_list]))

	if not(len(road_xy_list) == 2 and len(road_xy_list[0]) == 2 and len(road_xy_list[1]) == 2):
		return json_input_number*1000+51	# road xy points data error
	
	elif road_isnumeric_check != 4:
		return json_input_number*1000+52	# road xy points number error

	##############################################################
	## check for delta T simulation inputs
	##############################################################
	# t_step data
	t_step_data_isnumeric_check = int(sum([isinstance(t_step, (int,float))] + [isinstance(t_max, (int,float))]))
	if t_step_data_isnumeric_check != 2:
		return json_input_number*1000+60		# time data number error
	
	elif t_step <= 0:
		return json_input_number*1000+61		# t_step value error
	
	elif t_max <= 0:	
		return json_input_number*1000+62		# t_max value error

	elif t_step >= t_max:
		return json_input_number*1000+63		# t_max and t_step inequality error

	##############################################################
	## check for part number
	##############################################################
	if json_input_data["particle number per cell"] is not None:
		if isinstance(json_input_data["particle number per cell"], int) == False:
			if round(np.sqrt(json_input_data["particle number per cell"])) - np.sqrt(json_input_data["particle number per cell"]) <= 1e-5: # square number check
				return json_input_number*1000+70		# particle number number error

	##############################################################
	## initial velocity
	##############################################################
	if initial_velocity[0] != None:
		if isinstance(initial_velocity[0], (int,float)) == False:
			return json_input_number*1000+80	# initial velocity number error

	if initial_velocity[1] != None:
		if isinstance(initial_velocity[1], (int,float)) == False:
			return json_input_number*1000+80	# initial velocity number error

	##############################################################
	## material dictionary
	##############################################################
	try: 
		if isinstance(material, dict) == False:
			return json_input_number*1000+90	# material file format error
	except:
		return json_input_number*1000+90		# material file format error
			
	else:
		for mat_info in material.values():
			
			# resistance
			if isinstance(mat_info['f'], list) == False:
				return json_input_number*1000+91	# material resistance format error

			elif len(mat_info['f']) != 2:
				return json_input_number*1000+92	# material resistance data error

			elif not( isinstance(mat_info['f'][0], (int,float)) and isinstance(mat_info['f'][1], (int,float)) ):
				return json_input_number*1000+93	# material resistance number error

			# entrainment growth rate
			if isinstance(mat_info['Es'], (int,float)) == False:
				return json_input_number*1000+94	# material entrainment number error

			# density
			if isinstance(mat_info['density'], (int,float)) == False:
				return json_input_number*1000+95	# material density number error

			elif mat_info['density'] <= 0:
				return json_input_number*1000+96	# material density value error

			# internal friction angle
			if isinstance(mat_info['phi'], (int,float)) == False:
				return json_input_number*1000+97	# material phi number error

			elif mat_info['phi'] < 0 or mat_info['phi'] > 45:
				return json_input_number*1000+98	# material phi value error 

			# max erode depth
			if isinstance(mat_info['max erode depth'], (int,float)) == False:
				return json_input_number*1000+99	# material max erode depth number error

			elif mat_info['max erode depth'] < 0:
				return json_input_number*1000+99.5	# material max erode depth value error 

	###########################################################
	## initial SPH interpolation option
	###########################################################
	if isinstance(initial_SPH, bool) == False:
		return json_input_number*1000+800	# initial SPH interpolation value error

	##############################################################
	## gravity
	##############################################################
	if isinstance(g, (int,float)) == False:
		return json_input_number*1000+100	# gravity number error

	##############################################################
	## COR
	##############################################################
	# COR_p2p
	if isinstance(COR[0], (int,float)) == False:
		return json_input_number*1000+110	# COR_p2p number error

	elif COR[0] < 0 or COR[0] > 1:
		return json_input_number*1000+111	# COR_p2p value error

	# COR_p2w
	if isinstance(COR[1], (int,float)) == False:
		return json_input_number*1000+112	# COR_p2w number error

	elif COR[1] < 0 or COR[1] > 1:
		return json_input_number*1000+113	# COR_p2w value error

	##############################################################
	## check for interpolation method
	##############################################################
	if isinstance(interp_method, str) == False:
		return json_input_number*1000+120		# interpolation method string error

	else:
		interp_method_list = interp_method.split(" ")

		if len(interp_method_list) == 1:
			if interp_method not in ['linear', 'cubic']:
				return json_input_number*1000+121		# interpolation method option error
		else:
			if interp_method_list[0] not in ['OK', 'UK']: 
				return json_input_number*1000+121		# interpolation method option error
			else:
				if interp_method_list[1] not in ['linear', 'power', 'gaussian', 'spherical', 'exponential']:
					return json_input_number*1000+122	# kriging interpolation semi-variogram option error
				
	##############################################################
	## check for entrainment method
	##############################################################
	if isinstance(entrainment_model, str) == False:
		return json_input_number*1000+130		# entrainment model string error
	else:
		if entrainment_model not in ['Hungr', 'Er']: 
			return json_input_number*1000+131		# entrainment model option error

	##############################################################
	## check for boundary computing algorithm
	##############################################################
	if isinstance(cluster_boundary, str) == False:
		return json_input_number*1000+140		# boundary algorithm string error
	else:
		if cluster_boundary not in ['ConvexHull', 'ConcaveHull']: 
			return json_input_number*1000+141		# boundary algorithm option error

	##############################################################
	## Es_theta_var
	##############################################################
	if isinstance(Es_theta_var, (int,float)) == False:
		return json_input_number*1000+150	# Es_theta_var number error

	elif Es_theta_var < 0 or Es_theta_var > 90:
		return json_input_number*1000+151	# Es_theta_var value error

	##############################################################
	## pathway algorithm - local cell size
	##############################################################
	try: 
		if not isinstance(cell_size, tuple):
			return json_input_number*1000+160	# local cell size error
	except:
		return json_input_number*1000+160	# local cell size error

	if len(cell_size) != 2:
		return json_input_number*1000+161	# local cell size error

	cell_size_isnumeric_check = int(sum([isinstance(lcs, (int,float)) for lcs in cell_size]))
	if cell_size_isnumeric_check != len(cell_size):
		return json_input_number*1000+162	# local cell size number error

	cell_size_value_check = int(sum([(lcs > 0 and lcs%2 == 1) for lcs in cell_size]))
	if cell_size_value_check != len(cell_size):
		return json_input_number*1000+163	# local cell size value error

	##############################################################
	## l_dp_min
	##############################################################
	if isinstance(l_dp_min, (int,float)) == False:
		return json_input_number*1000+170	# l_dp_min number error

	elif l_dp_min <= 0:
		return json_input_number*1000+171	# l_dp_min value error

	##############################################################
	## check for Concave Hull options
	##############################################################
	if cluster_boundary == 'ConcaveHull':	# when ConcaveHull is used 
		if isinstance(concave_hull_parameter_dict, dict) == False: 
			return json_input_number*1000+180		# ConcaveHull algorithm parameter format error

		for cbi_key in concave_hull_parameter_dict.keys():
			if cbi_key not in ["max iteration", "max alpha", "min alpha"]: 
				return json_input_number*1000+181		# ConcaveHull algorithm parameter key error

			if isinstance(concave_hull_parameter_dict[cbi_key], (int, float)) == False: 
				return json_input_number*1000+182		# ConcaveHull algorithm parameter number error

		if concave_hull_parameter_dict["max iteration"] <= 0:
			return json_input_number*1000+183		# ConcaveHull algorithm max iteration value error

		if concave_hull_parameter_dict["min alpha"] < 0:
			return json_input_number*1000+184		# ConcaveHull algorithm min alpha value error

		if concave_hull_parameter_dict["max alpha"] <= 0 or concave_hull_parameter_dict["max alpha"] <= concave_hull_parameter_dict["min alpha"]:
			return json_input_number*1000+185		# ConcaveHull algorithm max alpha value error

	##############################################################
	## check for cluster merge criteria
	##############################################################
	if isinstance(merge_overlap_ratio, (int,float)) == False: 
		return json_input_number*1000+190		# merge overlap ratio number error
	elif merge_overlap_ratio < 0 or merge_overlap_ratio > 1:
		return json_input_number*1000+191		# merge overlap ratio value error

	##############################################################
	## cpu multiprocessing number
	##############################################################
	if not( isinstance(max_cpu_num, int) ):
		return json_input_number*1000+210		# cpu multiprocessing number error

	elif max_cpu_num < 1:
		return json_input_number*1000+211		# cpu multiprocessing value error
	
	##############################################################
	## check for optimal barrier location selection
	##############################################################

	if not ( isinstance(opt_total_loop, int) and isinstance(opt_iter_max, int) and isinstance(opt_find_iter_limit, int) ):
		return json_input_number*1000+300		# optimal barrier location loop number error

	if not(opt_barrier_num_limit == [None, None]): 
		if not( isinstance(opt_barrier_num_limit[0], int) and isinstance(opt_barrier_num_limit[1], int) ):
			return json_input_number*1000+301		# optimal barrier location limit number error

		if opt_barrier_num_limit[0] > opt_barrier_num_limit[1] or opt_barrier_num_limit[0] < 0 or opt_barrier_num_limit[1] < 0:
			return json_input_number*1000+302		# optimal barrier location limit value error

	if isinstance(opt_avoid_dfl, bool) == False:
		return json_input_number*1000+303		# optimal barrier location avoid DFL option error

	# for closed-type barrier location selection
	# optimization weighting factor
	if int(sum([isinstance(owc, (int,float)) for owc in opt_weight_close])) != len(opt_weight_close):
		return json_input_number*1000+310		# optimal closed barrier number error
	elif round(sum(opt_weight_close)) != 1:
		return json_input_number*1000+311		# optimal closed barrier sum value error

	if isinstance(output_optimal_step, bool) == False:
		return json_input_number*1000+350		# output_optimal_step option error

	##############################################################
	## plotting
	##############################################################
	# check boolean type
	if not (isinstance(plot_map_2D, bool) and isinstance(plot_animation_2D, bool) and isinstance(plot_map_3D, bool) and isinstance(plot_animation_3D, bool) and isinstance(open_plot, bool) and isinstance(plot_animation_2D_boundary, bool) ):
		return json_input_number*1000+400		# figure option error

	if plot_map_2D:
		if int(sum([isinstance(p2D, (int,float)) for p2D in plot_2D_max_limits])) != 9:
			return json_input_number*1000+410		# 2D plot option error

		elif int(sum([p2D > 0 for p2D in plot_2D_max_limits])) != 9:
			return json_input_number*1000+411		# 2D plot option value error
		
	if plot_map_3D:
		if int(sum([isinstance(p3D, (int,float)) for p3D in plot_3D_max_limits] + [isinstance(plot_3D_z_offset, (int,float))])) != 10:
			return json_input_number*1000+420		# 3D plot option error

		elif int(sum([p3D > 0 for p3D in plot_3D_max_limits])) != 9 or plot_3D_z_offset < 0:
			return json_input_number*1000+421		# 3D plot option value error

	if plot_animation_2D or plot_animation_3D:
		if int(sum([isinstance(aniD, (int,float)) for aniD in animation] + [isinstance(animation_3D_z_offset, (int,float))])) != 4:
			return json_input_number*1000+430		# animation option error

		elif int(sum([aniD > 0 for aniD in animation])) != 3 or animation_3D_z_offset < 0:
			return json_input_number*1000+431		# animation option value error

	if not ( isinstance(marker_size, (int,float)) and isinstance(line_width, (int,float)) and isinstance(layout_width, (int,float)) and isinstance(layout_height, (int,float)) ):
		return json_input_number*1000+440		# figure size error
	
	elif (marker_size <= 0) or (line_width <= 0) or (layout_width <= 0) or (layout_height <= 0):
		return json_input_number*1000+441		# figure size value error

	##############################################################
	## output
	##############################################################
	# output options
	if isinstance(csv_output, bool) == False:
		return json_input_number*1000+500		# csv output option error

	# decimal point number
	if isinstance(dp, int) == False:
		return json_input_number*1000+510		# decimal point number error

	elif dp < 0:
		return json_input_number*1000+511		# decimal point value error

	return 0	# no error in the json input file

# SPEC-debris-combined
def check_SPEC_debris_combined_json_input_v11_00(json_input_number, json_file_name, json_input_data):

	##################################################################################################################
	## read json file and extract input file data - also check whether JSON input file has no issues
	##################################################################################################################

	try: # exception for json input file error

		###########################################################
		## Name and folder information
		###########################################################
		# project name / export name
		exportName = json_input_data["project name"] 

		# project folder path
		folder_path = json_input_data["folder path"]
		if folder_path == None:  # current multiple json file folder path == each simulation folder path
			folder_path = os.path.dirname(json_file_name)+'/'
		else:
			folder_path = folder_path+'/'

		###########################################################
		## map data - DEM, road, wall/building
		###########################################################
		# DEM data
		flowpath_file_name = json_input_data["flowpath file name"]
		source_file_name = json_input_data["source file name"]
		material_file_name = json_input_data["material file name"] 

		# road and goal XY-coordinates
		road_xy_list = [tuple(xy_pt) for xy_pt in json_input_data["road xy"]]

		###########################################################
		## SPEC-debris analysis set-up and material properties
		###########################################################
		# time step parameters
		t_step = json_input_data["time step interval"]
		t_max = json_input_data["maximum simulation time"]

		# particle number
		if json_input_data["particle number per cell"] is None:
			part_num_per_cell = 1
		else:
			try:
				part_num_per_cell = int(json_input_data["particle number per cell"])
			except:
				part_num_per_cell = 1

		# initial setup of debris-flow - initial velocity
		initial_velocity = (json_input_data["initial u_x"], json_input_data["initial u_y"])

		# material
		material = {}
		for mat_key, mat_info in json_input_data["material"].items():
			material[int(mat_key)] = mat_info

		# load - gravity
		g = json_input_data["gravitational acceleration"]

		# interaction - coefficient of restitution COR
		COR = (json_input_data["coefficient of restitution (COR)"]["particle with particle"], 
				json_input_data["coefficient of restitution (COR)"]["particle with wall"])
		# COR = (1.0, json_input_data["coefficient of restitution (COR)"]["particle with wall"]) 

		# perform SPH interpolation at time step = 0
		initial_SPH = json_input_data["initial SPH"]

		###########################################################
		## Algorithm options 
		###########################################################
		# interpolation method
		interp_method = json_input_data["interpolation method"]

		# Entrainment model
		entrainment_model = json_input_data["entrainment model"]

		# cluster boundary algorithm
		cluster_boundary = json_input_data["cluster boundary method"]	

		###########################################################
		## Control parameters 
		###########################################################
		# no entrainment when climbing wall or free-falling
		Es_theta_var = json_input_data["free fall angle variation"]

		# simulation coefficients and constants
		cell_size = tuple(json_input_data["local cell sizes"]) 		

		# SPH smoothing length computation
		l_dp_min = json_input_data["l_dp_min"]

		# cluster boundary control parameters for Concave-Hull
		concave_hull_parameter_dict = json_input_data["concave hull algorithm parameter"]

		# mutliple cluster merge
		merge_overlap_ratio = json_input_data["merge overlap ratio"]

		# multiprocessing
		max_cpu_num = json_input_data["max cpu num"]

		###########################################################
		## optimal barrier location options - closed-type only
		###########################################################

		# selection of optimal barrier parameters 
		opt_total_loop = json_input_data["optimal barrier location selection option"]["optimal total loop"]
		opt_avoid_dfl = json_input_data["optimal barrier location selection option"]["optimal avoid dfl"]
		opt_iter_max = json_input_data["optimal barrier location selection option"]["optimal iteration max"]
		opt_find_iter_limit = json_input_data["optimal barrier location selection option"]["optimal find iteration limit"]
		opt_barrier_num_limit = [json_input_data["optimal barrier location selection option"]["optimal barrier num min"],
									json_input_data["optimal barrier location selection option"]["optimal barrier num max"]]

		# track cost per epoch
		output_optimal_step = json_input_data["output optimal step"]

		# mixed both open and closed-type barrier
		opt_weight_combined = [
			json_input_data["combined barrier optimization criteria"]["volume(V)"],
			json_input_data["combined barrier optimization criteria"]["pressure(P)"],
			json_input_data["combined barrier optimization criteria"]["distance_from_road(D)"],
			json_input_data["combined barrier optimization criteria"]["closed_barrier_number(NC)"],
			json_input_data["combined barrier optimization criteria"]["opened_barrier_number(NO)"]
		]

		# assumed barrier performance for optimal opened-type barriers
		open_performance = [
			json_input_data["open barrier performance"]["speed_ratio(SR)"],
			json_input_data["open barrier performance"]["trap_ratio(TR)"]
		]

		# vulnerability index
		min_uV = [0.0, 0.0]
		VI_crit = json_input_data["critical vulnerability index (VI)"] 
		RC_bool = json_input_data["reinforced concrete (RC) wall"]
		
		###########################################################
		## plotting options 
		###########################################################

		# plotting options
		plot_map_2D = json_input_data["static plot option"]["plot map 2D"]
		plot_map_3D = json_input_data["static plot option"]["plot map 3D"]
				
		# animation options
		plot_animation_2D = json_input_data["animation option"]["plot animation 2D"]
		plot_animation_3D = json_input_data["animation option"]["plot animation 3D"]
		plot_animation_2D_boundary = json_input_data["animation option"]["plot animation 2D boundary"]

		# open plot after creating
		open_plot = json_input_data["open plot"]

		# 3D plot option
		plot_3D_z_offset = json_input_data["static plot option"]["plot 3D z offset"]

		# animation options
		animation = [
			json_input_data["animation option"]["frame duration"], 
			json_input_data["animation option"]["frame transition"], 
			json_input_data["animation option"]["contour elevation interval"]
		]
		animation_3D_z_offset = json_input_data["animation option"]["animation 3D z offset"]

		# plot maximum legend
		plot_2D_max_limits = [
			json_input_data["max parameter legend"]["cluster max velocity(u)"],
			json_input_data["max parameter legend"]["cluster max depth(h)"],
			json_input_data["max parameter legend"]["cluster max volume(V)"],
			json_input_data["max parameter legend"]["cluster max distance_from_road(D)"],
			json_input_data["max parameter legend"]["cluster max pressure(P)"],
			json_input_data["max parameter legend"]["particle max velocity(u)"],
			json_input_data["max parameter legend"]["particle max depth(h)"],
			json_input_data["max parameter legend"]["particle max volume(V)"],
			json_input_data["max parameter legend"]["particle max pressure(P)"]
		]
		plot_3D_max_limits = deepcopy(plot_2D_max_limits)

		# marker, line, and plot size options
		marker_size = json_input_data["marker size"]
		line_width = json_input_data["line width"]
		layout_width = json_input_data["layout width"]
		layout_height = json_input_data["layout height"]

		###########################################################
		## output options 
		###########################################################
		# csv data export for SPEC-debris analysis
		csv_output = json_input_data["csv output"]

		# data decimal places
		dp = json_input_data["decimal points"]

	except (KeyError, NameError):
		return json_input_number*1000+2	# input file variable error

	##################################################################################################################
	## check valid input file type, existance, and value
	##################################################################################################################

	##########################################################
	## flowpath_file_name
	##########################################################
	# check if it exists or has valid naming
	if os.path.isfile(folder_path+flowpath_file_name) == False:
		return json_input_number*1000+20	# flowpath file error

	# check file type
	flowpath_file_name_list = flowpath_file_name.split('.')
	flowpath_file_name_type = flowpath_file_name_list[-1]
	if flowpath_file_name_type not in ['csv', 'las', 'grd']:
		return json_input_number*1000+21	# flowpath filetype error

	##########################################################
	## source_file_name
	##########################################################
	# check if it exists or has valid naming
	if os.path.isfile(folder_path+source_file_name) == False:
		return json_input_number*1000+30	# source file error

	# check file type
	source_file_name_list = source_file_name.split('.')
	source_file_name_type = source_file_name_list[-1]
	if source_file_name_type not in ['csv', 'las', 'grd']:
		return json_input_number*1000+31	# source filetype error

	##############################################################
	## material_file_name
	##############################################################
	# import material file if number of number of material is higher than 1
	if material_file_name is not None and len(material.keys()) > 1:

		# check if it exists or has valid naming
		if os.path.isfile(folder_path+material_file_name) == False:
			return json_input_number*1000+40	# material file error

		# check file type
		material_file_name_list = material_file_name.split('.')
		material_file_name_type = material_file_name_list[-1]
		if material_file_name_type not in ['csv', 'las', 'grd']:
			return json_input_number*1000+41	# material filetype error

	# missing material data 
	elif material_file_name is None and len(material.keys()) > 1:
		return json_input_number*1000+42	# material file required

	##############################################################
	## road
	##############################################################
	# road format test
	try: 
		if not isinstance(road_xy_list, list):
			return json_input_number*1000+50	# road xy points error
	except:
		return json_input_number*1000+50		# road xy points error

	road_isnumeric_check = int(sum([isinstance(rxy[0], (int,float)) + isinstance(rxy[1], (int,float)) for rxy in road_xy_list]))

	if not(len(road_xy_list) == 2 and len(road_xy_list[0]) == 2 and len(road_xy_list[1]) == 2):
		return json_input_number*1000+51	# road xy points data error
	
	elif road_isnumeric_check != 4:
		return json_input_number*1000+52	# road xy points number error

	##############################################################
	## check for delta T simulation inputs
	##############################################################
	# t_step data
	t_step_data_isnumeric_check = int(sum([isinstance(t_step, (int,float))] + [isinstance(t_max, (int,float))]))
	if t_step_data_isnumeric_check != 2:
		return json_input_number*1000+60		# time data number error
	
	elif t_step <= 0:
		return json_input_number*1000+61		# t_step value error
	
	elif t_max <= 0:	
		return json_input_number*1000+62		# t_max value error

	elif t_step >= t_max:
		return json_input_number*1000+63		# t_max and t_step inequality error

	##############################################################
	## check for part number
	##############################################################
	if json_input_data["particle number per cell"] is not None:
		if isinstance(json_input_data["particle number per cell"], int) == False:
			if round(np.sqrt(json_input_data["particle number per cell"])) - np.sqrt(json_input_data["particle number per cell"]) <= 1e-5: # square number check
				return json_input_number*1000+70		# particle number number error

	##############################################################
	## initial velocity
	##############################################################
	if initial_velocity[0] != None:
		if isinstance(initial_velocity[0], (int,float)) == False:
			return json_input_number*1000+80	# initial velocity number error

	if initial_velocity[1] != None:
		if isinstance(initial_velocity[1], (int,float)) == False:
			return json_input_number*1000+80	# initial velocity number error

	##############################################################
	## material dictionary
	##############################################################
	try: 
		if isinstance(material, dict) == False:
			return json_input_number*1000+90	# material file format error
	except:
		return json_input_number*1000+90		# material file format error
			
	else:
		for mat_info in material.values():
			
			# resistance
			if isinstance(mat_info['f'], list) == False:
				return json_input_number*1000+91	# material resistance format error

			elif len(mat_info['f']) != 2:
				return json_input_number*1000+92	# material resistance data error

			elif not( isinstance(mat_info['f'][0], (int,float)) and isinstance(mat_info['f'][1], (int,float)) ):
				return json_input_number*1000+93	# material resistance number error

			# entrainment growth rate
			if isinstance(mat_info['Es'], (int,float)) == False:
				return json_input_number*1000+94	# material entrainment number error

			# density
			if isinstance(mat_info['density'], (int,float)) == False:
				return json_input_number*1000+95	# material density number error

			elif mat_info['density'] <= 0:
				return json_input_number*1000+96	# material density value error

			# internal friction angle
			if isinstance(mat_info['phi'], (int,float)) == False:
				return json_input_number*1000+97	# material phi number error

			elif mat_info['phi'] < 0 or mat_info['phi'] > 45:
				return json_input_number*1000+98	# material phi value error 

			# max erode depth
			if isinstance(mat_info['max erode depth'], (int,float)) == False:
				return json_input_number*1000+99	# material max erode depth number error

			elif mat_info['max erode depth'] < 0:
				return json_input_number*1000+99.5	# material max erode depth value error 

	###########################################################
	## initial SPH interpolation option
	###########################################################
	if isinstance(initial_SPH, bool) == False:
		return json_input_number*1000+800	# initial SPH interpolation value error

	##############################################################
	## gravity
	##############################################################
	if isinstance(g, (int,float)) == False:
		return json_input_number*1000+100	# gravity number error

	##############################################################
	## COR
	##############################################################
	# COR_p2p
	if isinstance(COR[0], (int,float)) == False:
		return json_input_number*1000+110	# COR_p2p number error

	elif COR[0] < 0 or COR[0] > 1:
		return json_input_number*1000+111	# COR_p2p value error

	# COR_p2w
	if isinstance(COR[1], (int,float)) == False:
		return json_input_number*1000+112	# COR_p2w number error

	elif COR[1] < 0 or COR[1] > 1:
		return json_input_number*1000+113	# COR_p2w value error

	##############################################################
	## check for interpolation method
	##############################################################
	if isinstance(interp_method, str) == False:
		return json_input_number*1000+120		# interpolation method string error

	else:
		interp_method_list = interp_method.split(" ")

		if len(interp_method_list) == 1:
			if interp_method not in ['linear', 'cubic']:
				return json_input_number*1000+121		# interpolation method option error
		else:
			if interp_method_list[0] not in ['OK', 'UK']: 
				return json_input_number*1000+121		# interpolation method option error
			else:
				if interp_method_list[1] not in ['linear', 'power', 'gaussian', 'spherical', 'exponential']:
					return json_input_number*1000+122	# kriging interpolation semi-variogram option error
				
	##############################################################
	## check for entrainment method
	##############################################################
	if isinstance(entrainment_model, str) == False:
		return json_input_number*1000+130		# entrainment model string error
	else:
		if entrainment_model not in ['Hungr', 'Er']: 
			return json_input_number*1000+131		# entrainment model option error

	##############################################################
	## check for boundary computing algorithm
	##############################################################
	if isinstance(cluster_boundary, str) == False:
		return json_input_number*1000+140		# boundary algorithm string error
	else:
		if cluster_boundary not in ['ConvexHull', 'ConcaveHull']: 
			return json_input_number*1000+141		# boundary algorithm option error

	##############################################################
	## Es_theta_var
	##############################################################
	if isinstance(Es_theta_var, (int,float)) == False:
		return json_input_number*1000+150	# Es_theta_var number error

	elif Es_theta_var < 0 or Es_theta_var > 90:
		return json_input_number*1000+151	# Es_theta_var value error

	##############################################################
	## pathway algorithm - local cell size
	##############################################################
	try: 
		if not isinstance(cell_size, tuple):
			return json_input_number*1000+160	# local cell size error
	except:
		return json_input_number*1000+160	# local cell size error

	if len(cell_size) != 2:
		return json_input_number*1000+161	# local cell size error

	cell_size_isnumeric_check = int(sum([isinstance(lcs, (int,float)) for lcs in cell_size]))
	if cell_size_isnumeric_check != len(cell_size):
		return json_input_number*1000+162	# local cell size number error

	cell_size_value_check = int(sum([(lcs > 0 and lcs%2 == 1) for lcs in cell_size]))
	if cell_size_value_check != len(cell_size):
		return json_input_number*1000+163	# local cell size value error

	##############################################################
	## l_dp_min
	##############################################################
	if isinstance(l_dp_min, (int,float)) == False:
		return json_input_number*1000+170	# l_dp_min number error

	elif l_dp_min <= 0:
		return json_input_number*1000+171	# l_dp_min value error

	##############################################################
	## check for Concave Hull options
	##############################################################
	if cluster_boundary == 'ConcaveHull':	# when ConcaveHull is used 
		if isinstance(concave_hull_parameter_dict, dict) == False: 
			return json_input_number*1000+180		# ConcaveHull algorithm parameter format error

		for cbi_key in concave_hull_parameter_dict.keys():
			if cbi_key not in ["max iteration", "max alpha", "min alpha"]: 
				return json_input_number*1000+181		# ConcaveHull algorithm parameter key error

			if isinstance(concave_hull_parameter_dict[cbi_key], (int, float)) == False: 
				return json_input_number*1000+182		# ConcaveHull algorithm parameter number error

		if concave_hull_parameter_dict["max iteration"] <= 0:
			return json_input_number*1000+183		# ConcaveHull algorithm max iteration value error

		if concave_hull_parameter_dict["min alpha"] < 0:
			return json_input_number*1000+184		# ConcaveHull algorithm min alpha value error

		if concave_hull_parameter_dict["max alpha"] <= 0 or concave_hull_parameter_dict["max alpha"] <= concave_hull_parameter_dict["min alpha"]:
			return json_input_number*1000+185		# ConcaveHull algorithm max alpha value error

	##############################################################
	## check for cluster merge criteria
	##############################################################
	if isinstance(merge_overlap_ratio, (int,float)) == False: 
		return json_input_number*1000+190		# merge overlap ratio number error
	elif merge_overlap_ratio < 0 or merge_overlap_ratio > 1:
		return json_input_number*1000+191		# merge overlap ratio value error

	##############################################################
	## cpu multiprocessing number
	##############################################################
	if not( isinstance(max_cpu_num, int) ):
		return json_input_number*1000+210		# cpu multiprocessing number error

	elif max_cpu_num < 1:
		return json_input_number*1000+211		# cpu multiprocessing value error
	
	##############################################################
	## check for optimal barrier location selection
	##############################################################

	if not ( isinstance(opt_total_loop, int) and isinstance(opt_iter_max, int) and isinstance(opt_find_iter_limit, int) ):
		return json_input_number*1000+300		# optimal barrier location loop number error

	if not(opt_barrier_num_limit == [None, None]): 
		if not( isinstance(opt_barrier_num_limit[0], int) and isinstance(opt_barrier_num_limit[1], int) ):
			return json_input_number*1000+301		# optimal barrier location limit number error

		if opt_barrier_num_limit[0] > opt_barrier_num_limit[1] or opt_barrier_num_limit[0] < 0 or opt_barrier_num_limit[1] < 0:
			return json_input_number*1000+302		# optimal barrier location limit value error

	if isinstance(opt_avoid_dfl, bool) == False:
		return json_input_number*1000+303		# optimal barrier location avoid DFL option error

	# for combined closed and open-type barrier location selection
	# optimization weighting factor
	if int(sum([isinstance(owco, (int,float)) for owco in opt_weight_combined])) != len(opt_weight_combined):
		return json_input_number*1000+320		# optimal combined barrier number error
	elif round(sum(opt_weight_combined)) != 1:
		return json_input_number*1000+321		# optimal combined barrier sum value error

	# open type barrier performance 
	if int(sum([isinstance(opp, (int,float)) for opp in open_performance])) != 2:
		return json_input_number*1000+330		# open type barrier number error
	elif open_performance[0] < 0 or open_performance[0] > 1:
		return json_input_number*1000+331		# open type barrier SR value error
	elif open_performance[1] < 0 or open_performance[1] > 1:
		return json_input_number*1000+332		# open type barrier TR value error

	# vulnerability index
	if not (isinstance(min_uV[0], (int,float)) and isinstance(min_uV[1], (int,float)) and isinstance(VI_crit, (int,float)) and isinstance(RC_bool, bool)):
		return json_input_number*1000+340		# vulnerability analysis number error
	elif min_uV[0] < 0 or min_uV[1] < 0:
		return json_input_number*1000+341		# vulnerability analysis min_uV value error
	elif VI_crit < 0 or VI_crit > 1:
		return json_input_number*1000+342		# vulnerability analysis VI_crit value error

	if isinstance(output_optimal_step, bool) == False:
		return json_input_number*1000+350		# output_optimal_step option error

	##############################################################
	## plotting
	##############################################################
	# check boolean type
	if not (isinstance(plot_map_2D, bool) and isinstance(plot_animation_2D, bool) and isinstance(plot_map_3D, bool) and isinstance(plot_animation_3D, bool) and isinstance(open_plot, bool) and isinstance(plot_animation_2D_boundary, bool) ):
		return json_input_number*1000+400		# figure option error

	if plot_map_2D:
		if int(sum([isinstance(p2D, (int,float)) for p2D in plot_2D_max_limits])) != 9:
			return json_input_number*1000+410		# 2D plot option error

		elif int(sum([p2D > 0 for p2D in plot_2D_max_limits])) != 9:
			return json_input_number*1000+411		# 2D plot option value error
		
	if plot_map_3D:
		if int(sum([isinstance(p3D, (int,float)) for p3D in plot_3D_max_limits] + [isinstance(plot_3D_z_offset, (int,float))])) != 10:
			return json_input_number*1000+420		# 3D plot option error

		elif int(sum([p3D > 0 for p3D in plot_3D_max_limits])) != 9 or plot_3D_z_offset < 0:
			return json_input_number*1000+421		# 3D plot option value error

	if plot_animation_2D or plot_animation_3D:
		if int(sum([isinstance(aniD, (int,float)) for aniD in animation] + [isinstance(animation_3D_z_offset, (int,float))])) != 4:
			return json_input_number*1000+430		# animation option error

		elif int(sum([aniD > 0 for aniD in animation])) != 3 or animation_3D_z_offset < 0:
			return json_input_number*1000+431		# animation option value error

	if not ( isinstance(marker_size, (int,float)) and isinstance(line_width, (int,float)) and isinstance(layout_width, (int,float)) and isinstance(layout_height, (int,float)) ):
		return json_input_number*1000+440		# figure size error
	
	elif (marker_size <= 0) or (line_width <= 0) or (layout_width <= 0) or (layout_height <= 0):
		return json_input_number*1000+441		# figure size value error

	##############################################################
	## output
	##############################################################
	# output options
	if isinstance(csv_output, bool) == False:
		return json_input_number*1000+500		# csv output option error

	# decimal point number
	if isinstance(dp, int) == False:
		return json_input_number*1000+510		# decimal point number error

	elif dp < 0:
		return json_input_number*1000+511		# decimal point value error

	return 0	# no error in the json input file

# SPEC-debris (debris-flow simulation or barrier performance)
def check_SPEC_debris_json_input_v11_00(json_input_number, json_file_name, json_input_data, wall_keyword=False):

	##################################################################################################################
	## read json file and extract input file data - also check whether JSON input file has no issues
	##################################################################################################################

	try: # exception for json input file error

		###########################################################
		## Name and folder information
		###########################################################
		# project name / export name
		exportName = json_input_data["project name"] 

		# project folder path
		folder_path = json_input_data["folder path"]
		if folder_path == None:  # current multiple json file folder path == each simulation folder path
			folder_path = os.path.dirname(json_file_name)+'/'
		else:
			folder_path = folder_path+'/'

		###########################################################
		## map data - DEM and road
		###########################################################
		# DEM data
		flowpath_file_name = json_input_data["flowpath file name"]
		source_file_name = json_input_data["source file name"]
		material_file_name = json_input_data["material file name"] 

		# road and goal XY-coordinates
		road_xy_list = [tuple(xy_pt) for xy_pt in json_input_data["road xy"]]

		###########################################################
		## map data - wall/building
		###########################################################
		if json_input_data["wall info"] == None:
			wall_info = None
		else:
			wall_info = []
			for wall_group_num_str, wall_i_info in json_input_data["wall info"].items():

				# for any box-shaped barriers [closed or parallel (P) slit or V-shaped (V) slit or baffles]
				# wall_info -> [wall_group_id, type ('P' or 'V'), slit_ratio, wall_segment_number, wall_segment_oriP (-90 ~ 90), wall_oriP (-90 ~ 90), 
				# 	thickness, length, Z_opt (1~4), height/elevation, central_X_coord, central_Y_coord]
				if wall_i_info["wall type"] in ["P", "V"]:
					wall_info.append([
						int(wall_group_num_str),
						wall_i_info["wall type"], 
						wall_i_info["slit ratio"], 
						wall_i_info["number of wall segments"], 
						wall_i_info["orientation of wall segments (Polar)"], 
						wall_i_info["orientation of wall overall (Polar)"], 
						wall_i_info["wall thickness"], 
						wall_i_info["wall length"], 
						wall_i_info["wall elevation option"], 
						wall_i_info["elevation or height"], 
						wall_i_info["wall centroid X"], 
						wall_i_info["wall centroid Y"]
					])

				# circle-shaped barriers [circular baffles]
				# wall_info -> [wall_group_id, type ('C'), cylinder_number, wall_oriP (-90 ~ 90), radius, length, Z_opt (1~4), height/elevation, central_X_coord, central_Y_coord]
				elif wall_i_info["wall type"] == "C":
					wall_info.append([
						int(wall_group_num_str),
						wall_i_info["wall type"], 
						wall_i_info["number of cylinder segments"], 
						wall_i_info["orientation of wall overall (Polar)"], 
						wall_i_info["cylinder radius"], 
						wall_i_info["wall length"], 
						wall_i_info["wall elevation option"], 
						wall_i_info["elevation or height"], 
						wall_i_info["wall centroid X"], 
						wall_i_info["wall centroid Y"]
					])

				# for user defined shape barriers
				# wall_info -> [wall_group_id, Type('BD'), Z_opt, height/elevation, XY_list ] 
				elif wall_i_info["wall type"] == "BD":
					wall_info.append([
						int(wall_group_num_str),
						wall_i_info["wall type"], 
						wall_i_info["wall elevation option"], 
						wall_i_info["elevation or height"], 
						wall_i_info["wall XY points"]
					])

		###########################################################
		## SPEC-debris analysis set-up and material properties
		###########################################################
		# time step parameters
		t_step = json_input_data["time step interval"]
		t_max = json_input_data["maximum simulation time"]

		# particle number
		if json_input_data["particle number per cell"] is None:
			part_num_per_cell = 1
		else:
			try:
				part_num_per_cell = int(json_input_data["particle number per cell"])
			except:
				part_num_per_cell = 1

		# initial setup of debris-flow - initial velocity
		initial_velocity = (json_input_data["initial u_x"], json_input_data["initial u_y"])

		# material
		material = {}
		for mat_key, mat_info in json_input_data["material"].items():
			material[int(mat_key)] = mat_info

		# load - gravity
		g = json_input_data["gravitational acceleration"]

		# interaction - coefficient of restitution COR
		COR = (json_input_data["coefficient of restitution (COR)"]["particle with particle"], 
				json_input_data["coefficient of restitution (COR)"]["particle with wall"])
		# COR = (1.0, json_input_data["coefficient of restitution (COR)"]["particle with wall"]) 

		# perform SPH interpolation at time step = 0
		initial_SPH = json_input_data["initial SPH"]

		###########################################################
		## Algorithm options 
		###########################################################
		# interpolation method
		interp_method = json_input_data["interpolation method"]

		# Entrainment model
		entrainment_model = json_input_data["entrainment model"]

		# cluster boundary algorithm
		cluster_boundary = json_input_data["cluster boundary method"]


		###########################################################
		## Control parameters 
		###########################################################
		# no entrainment when climbing wall or free-falling
		Es_theta_var = json_input_data["free fall angle variation"]

		# simulation coefficients and constants
		cell_size = tuple(json_input_data["local cell sizes"]) 		

		# SPH smoothing length computation
		l_dp_min = json_input_data["l_dp_min"]

		# cluster boundary control parameters for Concave-Hull
		concave_hull_parameter_dict = json_input_data["concave hull algorithm parameter"]

		# mutliple cluster merge
		merge_overlap_ratio = json_input_data["merge overlap ratio"]

		# multiprocessing
		max_cpu_num = json_input_data["max cpu num"]

		
		###########################################################
		## plotting options 
		###########################################################
		
		# plotting options
		plot_map_2D = json_input_data["static plot option"]["plot map 2D"]
		plot_map_3D = json_input_data["static plot option"]["plot map 3D"]
				
		# animation options
		plot_animation_2D = json_input_data["animation option"]["plot animation 2D"]
		plot_animation_3D = json_input_data["animation option"]["plot animation 3D"]
		plot_animation_2D_boundary = json_input_data["animation option"]["plot animation 2D boundary"]

		# open plot after creating
		open_plot = json_input_data["open plot"]

		# 3D plot option
		plot_3D_z_offset = json_input_data["static plot option"]["plot 3D z offset"]

		# animation options
		animation = [
			json_input_data["animation option"]["frame duration"], 
			json_input_data["animation option"]["frame transition"], 
			json_input_data["animation option"]["contour elevation interval"]
		]
		animation_3D_z_offset = json_input_data["animation option"]["animation 3D z offset"]

		# plot maximum legend
		plot_2D_max_limits = [
			json_input_data["max parameter legend"]["cluster max velocity(u)"],
			json_input_data["max parameter legend"]["cluster max depth(h)"],
			json_input_data["max parameter legend"]["cluster max volume(V)"],
			json_input_data["max parameter legend"]["cluster max distance_from_road(D)"],
			json_input_data["max parameter legend"]["cluster max pressure(P)"],
			json_input_data["max parameter legend"]["particle max velocity(u)"],
			json_input_data["max parameter legend"]["particle max depth(h)"],
			json_input_data["max parameter legend"]["particle max volume(V)"],
			json_input_data["max parameter legend"]["particle max pressure(P)"]
		]
		plot_3D_max_limits = deepcopy(plot_2D_max_limits)

		# marker, line, and plot size options
		marker_size = json_input_data["marker size"]
		line_width = json_input_data["line width"]
		layout_width = json_input_data["layout width"]
		layout_height = json_input_data["layout height"]


		###########################################################
		## output options 
		###########################################################
		# csv data export for SPEC-debris analysis
		csv_output = json_input_data["csv output"]

		# data decimal places
		dp = json_input_data["decimal points"]

	except (KeyError, NameError):
		return json_input_number*1000+2	# input file variable error

	##################################################################################################################
	## check valid input file type, existance, and value
	##################################################################################################################

	##########################################################
	## flowpath_file_name
	##########################################################
	# check if it exists or has valid naming
	if os.path.isfile(folder_path+flowpath_file_name) == False:
		return json_input_number*1000+20	# flowpath file error

	# check file type
	flowpath_file_name_list = flowpath_file_name.split('.')
	flowpath_file_name_type = flowpath_file_name_list[-1]
	if flowpath_file_name_type not in ['csv', 'las', 'grd']:
		return json_input_number*1000+21	# flowpath filetype error

	##########################################################
	## source_file_name
	##########################################################
	# check if it exists or has valid naming
	if os.path.isfile(folder_path+source_file_name) == False:
		return json_input_number*1000+30	# source file error

	# check file type
	source_file_name_list = source_file_name.split('.')
	source_file_name_type = source_file_name_list[-1]
	if source_file_name_type not in ['csv', 'las', 'grd']:
		return json_input_number*1000+31	# source filetype error

	##############################################################
	## material_file_name
	##############################################################
	# import material file if number of number of material is higher than 1
	if material_file_name is not None and len(material.keys()) > 1:

		# check if it exists or has valid naming
		if os.path.isfile(folder_path+material_file_name) == False:
			return json_input_number*1000+40	# material file error

		# check file type
		material_file_name_list = material_file_name.split('.')
		material_file_name_type = material_file_name_list[-1]
		if material_file_name_type not in ['csv', 'las', 'grd']:
			return json_input_number*1000+41	# material filetype error

	# missing material data 
	elif material_file_name is None and len(material.keys()) > 1:
		return json_input_number*1000+42	# material file required

	##############################################################
	## road
	##############################################################
	# road format test
	try: 
		if not isinstance(road_xy_list, list):
			return json_input_number*1000+50	# road xy points error
	except:
		return json_input_number*1000+50		# road xy points error

	road_isnumeric_check = int(sum([isinstance(rxy[0], (int,float)) + isinstance(rxy[1], (int,float)) for rxy in road_xy_list]))

	if not(len(road_xy_list) == 2 and len(road_xy_list[0]) == 2 and len(road_xy_list[1]) == 2):
		return json_input_number*1000+51	# road xy points data error
	
	elif road_isnumeric_check != 4:
		return json_input_number*1000+52	# road xy points number error

	##############################################################
	## check for delta T simulation inputs
	##############################################################
	# t_step data
	t_step_data_isnumeric_check = int(sum([isinstance(t_step, (int,float))] + [isinstance(t_max, (int,float))]))
	if t_step_data_isnumeric_check != 2:
		return json_input_number*1000+60		# time data number error
	
	elif t_step <= 0:
		return json_input_number*1000+61		# t_step value error
	
	elif t_max <= 0:	
		return json_input_number*1000+62		# t_max value error

	elif t_step >= t_max:
		return json_input_number*1000+63		# t_max and t_step inequality error

	##############################################################
	## check for part number
	##############################################################
	if json_input_data["particle number per cell"] is not None:
		if isinstance(json_input_data["particle number per cell"], int) == False:
			if round(np.sqrt(json_input_data["particle number per cell"])) - np.sqrt(json_input_data["particle number per cell"]) <= 1e-5: # square number check
				return json_input_number*1000+70		# particle number number error

	##############################################################
	## initial velocity
	##############################################################
	if initial_velocity[0] != None:
		if isinstance(initial_velocity[0], (int,float)) == False:
			return json_input_number*1000+80	# initial velocity number error

	if initial_velocity[1] != None:
		if isinstance(initial_velocity[1], (int,float)) == False:
			return json_input_number*1000+80	# initial velocity number error

	##############################################################
	## material dictionary
	##############################################################
	try: 
		if isinstance(material, dict) == False:
			return json_input_number*1000+90	# material file format error
	except:
		return json_input_number*1000+90		# material file format error
			
	else:
		for mat_info in material.values():
			
			# resistance
			if isinstance(mat_info['f'], list) == False:
				return json_input_number*1000+91	# material resistance format error

			elif len(mat_info['f']) != 2:
				return json_input_number*1000+92	# material resistance data error

			elif not( isinstance(mat_info['f'][0], (int,float)) and isinstance(mat_info['f'][1], (int,float)) ):
				return json_input_number*1000+93	# material resistance number error

			# entrainment growth rate
			if isinstance(mat_info['Es'], (int,float)) == False:
				return json_input_number*1000+94	# material entrainment number error

			# density
			if isinstance(mat_info['density'], (int,float)) == False:
				return json_input_number*1000+95	# material density number error

			elif mat_info['density'] <= 0:
				return json_input_number*1000+96	# material density value error

			# internal friction angle
			if isinstance(mat_info['phi'], (int,float)) == False:
				return json_input_number*1000+97	# material phi number error

			elif mat_info['phi'] < 0 or mat_info['phi'] > 45:
				return json_input_number*1000+98	# material phi value error 

			# max erode depth
			if isinstance(mat_info['max erode depth'], (int,float)) == False:
				return json_input_number*1000+99	# material max erode depth number error

			elif mat_info['max erode depth'] < 0:
				return json_input_number*1000+99.5	# material max erode depth value error 

	###########################################################
	## initial SPH interpolation option
	###########################################################
	if isinstance(initial_SPH, bool) == False:
		return json_input_number*1000+800	# initial SPH interpolation value error

	##############################################################
	## gravity
	##############################################################
	if isinstance(g, (int,float)) == False:
		return json_input_number*1000+100	# gravity number error

	##############################################################
	## COR
	##############################################################
	# COR_p2p
	if isinstance(COR[0], (int,float)) == False:
		return json_input_number*1000+110	# COR_p2p number error

	elif COR[0] < 0 or COR[0] > 1:
		return json_input_number*1000+111	# COR_p2p value error

	# COR_p2w
	if isinstance(COR[1], (int,float)) == False:
		return json_input_number*1000+112	# COR_p2w number error

	elif COR[1] < 0 or COR[1] > 1:
		return json_input_number*1000+113	# COR_p2w value error

	##############################################################
	## check for interpolation method
	##############################################################
	if isinstance(interp_method, str) == False:
		return json_input_number*1000+120		# interpolation method string error

	else:
		interp_method_list = interp_method.split(" ")

		if len(interp_method_list) == 1:
			if interp_method not in ['linear', 'cubic']:
				return json_input_number*1000+121		# interpolation method option error
		else:
			if interp_method_list[0] not in ['OK', 'UK']: 
				return json_input_number*1000+121		# interpolation method option error
			else:
				if interp_method_list[1] not in ['linear', 'power', 'gaussian', 'spherical', 'exponential']:
					return json_input_number*1000+122	# kriging interpolation semi-variogram option error
				
	##############################################################
	## check for entrainment method
	##############################################################
	if isinstance(entrainment_model, str) == False:
		return json_input_number*1000+130		# entrainment model string error
	else:
		if entrainment_model not in ['Hungr', 'Er']: 
			return json_input_number*1000+131		# entrainment model option error

	##############################################################
	## check for boundary computing algorithm
	##############################################################
	if isinstance(cluster_boundary, str) == False:
		return json_input_number*1000+140		# boundary algorithm string error
	else:
		if cluster_boundary not in ['ConvexHull', 'ConcaveHull']: 
			return json_input_number*1000+141		# boundary algorithm option error

	##############################################################
	## Es_theta_var
	##############################################################
	if isinstance(Es_theta_var, (int,float)) == False:
		return json_input_number*1000+150	# Es_theta_var number error

	elif Es_theta_var < 0 or Es_theta_var > 90:
		return json_input_number*1000+151	# Es_theta_var value error

	##############################################################
	## pathway algorithm - local cell size
	##############################################################
	try: 
		if not isinstance(cell_size, tuple):
			return json_input_number*1000+160	# local cell size error
	except:
		return json_input_number*1000+160	# local cell size error

	if len(cell_size) != 2:
		return json_input_number*1000+161	# local cell size error

	cell_size_isnumeric_check = int(sum([isinstance(lcs, (int,float)) for lcs in cell_size]))
	if cell_size_isnumeric_check != len(cell_size):
		return json_input_number*1000+162	# local cell size number error

	cell_size_value_check = int(sum([(lcs > 0 and lcs%2 == 1) for lcs in cell_size]))
	if cell_size_value_check != len(cell_size):
		return json_input_number*1000+163	# local cell size value error

	##############################################################
	## l_dp_min
	##############################################################
	if isinstance(l_dp_min, (int,float)) == False:
		return json_input_number*1000+170	# l_dp_min number error

	elif l_dp_min <= 0:
		return json_input_number*1000+171	# l_dp_min value error

	##############################################################
	## check for Concave Hull options
	##############################################################
	if cluster_boundary == 'ConcaveHull':	# when ConcaveHull is used 
		if isinstance(concave_hull_parameter_dict, dict) == False: 
			return json_input_number*1000+180		# ConcaveHull algorithm parameter format error

		for cbi_key in concave_hull_parameter_dict.keys():
			if cbi_key not in ["max iteration", "max alpha", "min alpha"]: 
				return json_input_number*1000+181		# ConcaveHull algorithm parameter key error

			if isinstance(concave_hull_parameter_dict[cbi_key], (int, float)) == False: 
				return json_input_number*1000+182		# ConcaveHull algorithm parameter number error

		if concave_hull_parameter_dict["max iteration"] <= 0:
			return json_input_number*1000+183		# ConcaveHull algorithm max iteration value error

		if concave_hull_parameter_dict["min alpha"] < 0:
			return json_input_number*1000+184		# ConcaveHull algorithm min alpha value error

		if concave_hull_parameter_dict["max alpha"] <= 0 or concave_hull_parameter_dict["max alpha"] <= concave_hull_parameter_dict["min alpha"]:
			return json_input_number*1000+185		# ConcaveHull algorithm max alpha value error

	##############################################################
	## check for cluster merge criteria
	##############################################################
	if isinstance(merge_overlap_ratio, (int,float)) == False: 
		return json_input_number*1000+190		# merge overlap ratio number error
	elif merge_overlap_ratio < 0 or merge_overlap_ratio > 1:
		return json_input_number*1000+191		# merge overlap ratio value error

	##############################################################
	## cpu multiprocessing number
	##############################################################
	if not( isinstance(max_cpu_num, int) ):
		return json_input_number*1000+210		# cpu multiprocessing number error

	elif max_cpu_num < 1:
		return json_input_number*1000+211		# cpu multiprocessing value error

	##############################################################
	## plotting
	##############################################################
	# check boolean type
	if not (isinstance(plot_map_2D, bool) and isinstance(plot_animation_2D, bool) and isinstance(plot_map_3D, bool) and isinstance(plot_animation_3D, bool) and isinstance(open_plot, bool) and isinstance(plot_animation_2D_boundary, bool) ):
		return json_input_number*1000+400		# figure option error

	if plot_map_2D:
		if int(sum([isinstance(p2D, (int,float)) for p2D in plot_2D_max_limits])) != 9:
			return json_input_number*1000+410		# 2D plot option error

		elif int(sum([p2D > 0 for p2D in plot_2D_max_limits])) != 9:
			return json_input_number*1000+411		# 2D plot option value error
		
	if plot_map_3D:
		if int(sum([isinstance(p3D, (int,float)) for p3D in plot_3D_max_limits] + [isinstance(plot_3D_z_offset, (int,float))])) != 10:
			return json_input_number*1000+420		# 3D plot option error

		elif int(sum([p3D > 0 for p3D in plot_3D_max_limits])) != 9 or plot_3D_z_offset < 0:
			return json_input_number*1000+421		# 3D plot option value error

	if plot_animation_2D or plot_animation_3D:
		if int(sum([isinstance(aniD, (int,float)) for aniD in animation] + [isinstance(animation_3D_z_offset, (int,float))])) != 4:
			return json_input_number*1000+430		# animation option error

		elif int(sum([aniD > 0 for aniD in animation])) != 3 or animation_3D_z_offset < 0:
			return json_input_number*1000+431		# animation option value error

	if not ( isinstance(marker_size, (int,float)) and isinstance(line_width, (int,float)) and isinstance(layout_width, (int,float)) and isinstance(layout_height, (int,float)) ):
		return json_input_number*1000+440		# figure size error
	
	elif (marker_size <= 0) or (line_width <= 0) or (layout_width <= 0) or (layout_height <= 0):
		return json_input_number*1000+441		# figure size value error

	##############################################################
	## output
	##############################################################
	# output options
	if isinstance(csv_output, bool) == False:
		return json_input_number*1000+500		# csv output option error

	# decimal point number
	if isinstance(dp, int) == False:
		return json_input_number*1000+510		# decimal point number error

	elif dp < 0:
		return json_input_number*1000+511		# decimal point value error

	##############################################################
	## wall info
	##############################################################
	if wall_info is not None or wall_keyword:

		if wall_keyword:
			if wall_info is None:
				return json_input_number*1000+600		# wall data error

		for wall_info_list in wall_info:

			# wall type
			if wall_info_list[1] not in ["P", "V", "C", "BD"]:
				return json_input_number*1000+601		# wall type option error

			# slit-type wall
			if wall_info_list[1] in ["P", "V"]:

				if int(sum([isinstance(w_dd, (int,float)) for w_dd in wall_info_list[2:]])) != 10:
					return json_input_number*1000+610		# slit-type wall parameter number error

				if wall_info_list[2] < 0 or wall_info_list[2] > 1: # slit ratio 
					return json_input_number*1000+611		# slit-type wall: slit ratio value error

				if wall_info_list[3] < 1 or isinstance(wall_info_list[3],int)==False:  # wall segment numbers
					return json_input_number*1000+612		# slit-type wall: number of wall segments value error

				if abs(wall_info_list[4]) > 90 or abs(wall_info_list[5]) > 90:  # wall orientations
					return json_input_number*1000+613		# slit-type wall: orientation angles value error

				if wall_info_list[6] <= 0: # thickness
					return json_input_number*1000+614		# slit-type wall: thickness value error

				if wall_info_list[7] <= 0: # length
					return json_input_number*1000+615		# slit-type wall: length value error

				if wall_info_list[6] > wall_info_list[7]: # thickness vs length
					return json_input_number*1000+616		# slit-type wall: thickness and length inequality error

				if wall_info_list[8] not in [1,2,3,4]: # Z-option
					return json_input_number*1000+617		# slit-type wall: elevation option error

				if wall_info_list[9] <= 0: # elevation or height
					return json_input_number*1000+618		# slit-type wall: elevation or height value error

			# circular baffles
			elif wall_info_list[1] == "C":

				if int(sum([isinstance(w_dd, (int,float)) for w_dd in wall_info_list[2:]])) != 8:
					return json_input_number*1000+620		# circular baffles parameter number error

				if wall_info_list[2] <= 1 or isinstance(wall_info_list[2],int)==False:  # cylinder numbers
					return json_input_number*1000+621		# circular baffles: number of cylinder value error

				if abs(wall_info_list[3]) > 90:  # wall orientations
					return json_input_number*1000+622		# circular baffles: orientation angles value error

				if wall_info_list[4] <= 0: # radius (=0.5*thickness)
					return json_input_number*1000+623		# circular baffles: radius value error

				if wall_info_list[5] <= 0: # length
					return json_input_number*1000+624		# circular baffles: section length value error

				if 2*wall_info_list[4] > wall_info_list[5]: # thickness (2*R) vs length
					return json_input_number*1000+625		# circular baffles: thickness and length inequality error

				if wall_info_list[5] not in [1,2,3,4]: # Z-option
					return json_input_number*1000+626		# circular baffles: elevation option error

				if wall_info_list[6] <= 0: # elevation or height
					return json_input_number*1000+627		# circular baffles: elevation or height value error

			# circular baffles
			elif wall_info_list[1] == "BD":
				
				if isinstance(wall_info_list[2], (int,float)) == False or isinstance(wall_info_list[3], (int,float)) == False:
					return json_input_number*1000+630		# building parameter number error

				if wall_info_list[2] not in [1,2,3,4]: # Z-option
					return json_input_number*1000+631		# building: elevation option error

				if wall_info_list[3] <= 0: # elevation or height
					return json_input_number*1000+632		# building: elevation or height value error 

				if len(wall_info_list[4]) <= 2:
					return json_input_number*1000+633		# building: XY points data number error

				check_numerical_BD_xy = [(isinstance(XY[0], (int,float)) and isinstance(XY[1], (int,float))) for XY in wall_info_list[4]]
				if int(sum(check_numerical_BD_xy)) != len(wall_info_list[4]):
					return json_input_number*1000+634		# building: XY points number error

				
	return 0	# no error in the json input file

# optimal closed barriers 
def check_closed_json_input_v8_00(json_input_number, json_file_name, json_input_data):

	##################################################################################################################
	## read json file and extract input file data - also check whether JSON input file has no issues
	##################################################################################################################

	try: # exception for json input file error

		exportName = json_input_data["project name"]
		folder_path = json_input_data['folder path']
		flowpath_file_name = json_input_data['flowpath file name']
		
		road_xy_list = json_input_data['road xy']
		
		opt_total_loop = json_input_data["optimal barrier location selection option"]["optimal total loop"]
		opt_avoid_dfl = json_input_data["optimal barrier location selection option"]["optimal avoid dfl"]
		opt_iter_max = json_input_data["optimal barrier location selection option"]["optimal iteration max"]
		opt_find_iter_limit = json_input_data["optimal barrier location selection option"]["optimal find iteration limit"]
		opt_barrier_num_limit = [json_input_data["optimal barrier location selection option"]["optimal barrier num min"],
									json_input_data["optimal barrier location selection option"]["optimal barrier num max"]]
		
		opt_weight_close = [
			json_input_data["closed barrier optimization criteria"]["volume(V)"],
			json_input_data["closed barrier optimization criteria"]["pressure(P)"],
			json_input_data["closed barrier optimization criteria"]["distance_from_road(D)"],
			json_input_data["closed barrier optimization criteria"]["closed_barrier_number(N)"]
		]

		output_optimal_step = json_input_data['output optimal step']
		
		plot_map_2D = json_input_data["static plot option"]["plot map 2D"]

		marker_size = json_input_data['marker size']
		line_width = json_input_data['line width']
		layout_width = json_input_data['layout width']
		layout_height = json_input_data['layout height']
		
		dp = json_input_data['decimal points']

		cluster_list_raw = json_input_data['SPEC_debris analysis cluster data']

	except (KeyError, NameError):
		return json_input_number*1000+2	# input file variable error

	##################################################################################################################
	## check valid input file type, existance, and value
	##################################################################################################################
	###########################################################
	## folder information
	###########################################################
	# project folder path
	if folder_path == None:  # current multiple json file folder path == each simulation folder path
		folder_path = os.path.dirname(json_file_name)+'/'
	else:
		folder_path = folder_path+'/'

	##########################################################
	## flowpath_file_name
	##########################################################
	# check if it exists or has valid naming
	if os.path.isfile(folder_path+flowpath_file_name) == False:
		return json_input_number*1000+20	# flowpath file error

	# check file type
	flowpath_file_name_list = flowpath_file_name.split('.')
	flowpath_file_name_type = flowpath_file_name_list[-1]
	if flowpath_file_name_type not in ['csv', 'las', 'grd']:
		return json_input_number*1000+21	# flowpath filetype error

	##############################################################
	## road
	##############################################################
	# road format test
	try: 
		if not isinstance(road_xy_list, list):
			return json_input_number*1000+50	# road xy points error
	except:
		return json_input_number*1000+50		# road xy points error

	road_isnumeric_check = int(sum([isinstance(rxy[0], (int,float)) + isinstance(rxy[1], (int,float)) for rxy in road_xy_list]))

	if not(len(road_xy_list) == 2 and len(road_xy_list[0]) == 2 and len(road_xy_list[1]) == 2):
		return json_input_number*1000+51	# road xy points data error
	
	elif road_isnumeric_check != 4:
		return json_input_number*1000+52	# road xy points number error
	
	##############################################################
	## check for optimal barrier location selection
	##############################################################
	if not ( isinstance(opt_total_loop, int) and isinstance(opt_iter_max, int) and isinstance(opt_find_iter_limit, int) ):
		return json_input_number*1000+300		# optimal barrier location loop number error

	if not(opt_barrier_num_limit == [None, None]): 
		if not( isinstance(opt_barrier_num_limit[0], int) and isinstance(opt_barrier_num_limit[1], int) ):
			return json_input_number*1000+301		# optimal barrier location limit number error

		if opt_barrier_num_limit[0] > opt_barrier_num_limit[1] or opt_barrier_num_limit[0] < 0 or opt_barrier_num_limit[1] < 0:
			return json_input_number*1000+302		# optimal barrier location limit value error

	if isinstance(opt_avoid_dfl, bool) == False:
		return json_input_number*1000+303		# optimal barrier location avoid DFL option error

	# for closed-type barrier location selection
	# optimization weighting factor
	if int(sum([isinstance(owc, (int,float)) for owc in opt_weight_close])) != len(opt_weight_close):
		return json_input_number*1000+310		# optimal closed barrier number error
	elif round(sum(opt_weight_close)) != 1:
		return json_input_number*1000+311		# optimal closed barrier sum value error

	if isinstance(output_optimal_step, bool) == False:
		return json_input_number*1000+350		# output_optimal_step option error

	##############################################################
	## plotting
	##############################################################
	# check boolean type
	if isinstance(plot_map_2D, bool) == False:
		return json_input_number*1000+400		# figure option error

	if not ( isinstance(marker_size, (int,float)) and isinstance(line_width, (int,float)) and isinstance(layout_width, (int,float)) and isinstance(layout_height, (int,float)) ):
		return json_input_number*1000+440		# figure size error
	
	elif (marker_size <= 0) or (line_width <= 0) or (layout_width <= 0) or (layout_height <= 0):
		return json_input_number*1000+441		# figure size value error

	##############################################################
	## output
	##############################################################
	# decimal point number
	if isinstance(dp, int) == False:
		return json_input_number*1000+510		# decimal point number error
	elif dp < 0:
		return json_input_number*1000+511		# decimal point value error

	return 0	# no error in the json input file

# optimal opened and closed barriers
def check_combined_json_input_v8_00(json_input_number, json_file_name, json_input_data):

	##################################################################################################################
	## read json file and extract input file data - also check whether JSON input file has no issues
	##################################################################################################################

	try: # exception for json input file error

		## extract data from json file
		exportName = json_input_data["project name"]
		folder_path = json_input_data['folder path']
		flowpath_file_name = json_input_data['flowpath file name']

		road_xy_list = json_input_data['road xy']

		material = {}
		for mat_key, mat_info in json_input_data['material'].items():
			material[int(mat_key)] = mat_info

		MAT = np.array(json_input_data['material id array'])
		DEM_no_wall = np.array(json_input_data['flowpath z array'])
		gridUniqueX = json_input_data['grid x array']
		gridUniqueY = json_input_data['grid y array']
		deltaX = json_input_data['grid delta_x']
		deltaY = json_input_data['grid delta_y']
		fb_0 = json_input_data['overall_fb_0']

		opt_total_loop = json_input_data["optimal barrier location selection option"]["optimal total loop"]
		opt_avoid_dfl = json_input_data["optimal barrier location selection option"]["optimal avoid dfl"]
		opt_iter_max = json_input_data["optimal barrier location selection option"]["optimal iteration max"]
		opt_find_iter_limit = json_input_data["optimal barrier location selection option"]["optimal find iteration limit"]
		opt_barrier_num_limit = [json_input_data["optimal barrier location selection option"]["optimal barrier num min"],
									json_input_data["optimal barrier location selection option"]["optimal barrier num max"]]

		open_performance = [
			json_input_data["open barrier performance"]["speed_ratio(SR)"],
			json_input_data["open barrier performance"]["trap_ratio(TR)"]
		]
		opt_weight_combined = [
			json_input_data["combined barrier optimization criteria"]["volume(V)"],
			json_input_data["combined barrier optimization criteria"]["pressure(P)"],
			json_input_data["combined barrier optimization criteria"]["distance_from_road(D)"],
			json_input_data["combined barrier optimization criteria"]["closed_barrier_number(NC)"],
			json_input_data["combined barrier optimization criteria"]["opened_barrier_number(NO)"]
		]

		cell_size = max(json_input_data['local cell sizes'])
		g = json_input_data["gravitational acceleration"] 
		entrainment_model = json_input_data['entrainment model'] 
		Es_theta_var = json_input_data['free fall angle variation'] 
		interp_method = json_input_data['interpolation method'] 
		min_uV = [0.0, 0.0] 
		VI_crit = json_input_data['critical vulnerability index (VI)'] 
		RC_bool = json_input_data['reinforced concrete (RC) wall'] 

		output_optimal_step = json_input_data['output optimal step']
		
		plot_map_2D = json_input_data["static plot option"]["plot map 2D"]

		marker_size = json_input_data['marker size']
		line_width = json_input_data['line width']
		layout_width = json_input_data['layout width']
		layout_height = json_input_data['layout height']
		
		max_cpu_num = json_input_data["max cpu num"]

		dp = json_input_data['decimal points']

		# import json into class object
		cluster_list_raw = json_input_data['SPEC_debris analysis cluster data']

	except (KeyError, NameError):
		return json_input_number*1000+2	# input file variable error

	##################################################################################################################
	## check valid input file type, existance, and value
	##################################################################################################################
	###########################################################
	## folder information
	###########################################################
	# project folder path
	if folder_path == None:  # current multiple json file folder path == each simulation folder path
		folder_path = os.path.dirname(json_file_name)+'/'
	else:
		folder_path = folder_path+'/'

	##########################################################
	## flowpath_file_name
	##########################################################
	# check if it exists or has valid naming
	if os.path.isfile(folder_path+flowpath_file_name) == False:
		return json_input_number*1000+20	# flowpath file error

	# check file type
	flowpath_file_name_list = flowpath_file_name.split('.')
	flowpath_file_name_type = flowpath_file_name_list[-1]
	if flowpath_file_name_type not in ['csv', 'las', 'grd']:
		return json_input_number*1000+21	# flowpath filetype error

	##############################################################
	## road
	##############################################################
	# road format test
	try: 
		if not isinstance(road_xy_list, list):
			return json_input_number*1000+50	# road xy points error
	except:
		return json_input_number*1000+50		# road xy points error

	road_isnumeric_check = int(sum([isinstance(rxy[0], (int,float)) + isinstance(rxy[1], (int,float)) for rxy in road_xy_list]))

	if not(len(road_xy_list) == 2 and len(road_xy_list[0]) == 2 and len(road_xy_list[1]) == 2):
		return json_input_number*1000+51	# road xy points data error
	
	elif road_isnumeric_check != 4:
		return json_input_number*1000+52	# road xy points number error

	##############################################################
	## material dictionary
	##############################################################
	try: 
		if isinstance(material, dict) == False:
			return json_input_number*1000+90	# material file format error
	except:
		return json_input_number*1000+90		# material file format error
			
	else:
		for mat_info in material.values():
			
			# resistance
			if isinstance(mat_info['f'], list) == False:
				return json_input_number*1000+91	# material resistance format error

			elif len(mat_info['f']) != 2:
				return json_input_number*1000+92	# material resistance data error

			elif not( isinstance(mat_info['f'][0], (int,float)) and isinstance(mat_info['f'][1], (int,float)) ):
				return json_input_number*1000+93	# material resistance number error

			# entrainment growth rate
			if isinstance(mat_info['Es'], (int,float)) == False:
				return json_input_number*1000+94	# material entrainment number error

			# density
			if isinstance(mat_info['density'], (int,float)) == False:
				return json_input_number*1000+95	# material density number error

			elif mat_info['density'] <= 0:
				return json_input_number*1000+96	# material density value error

			# internal friction angle
			if isinstance(mat_info['phi'], (int,float)) == False:
				return json_input_number*1000+97	# material phi number error

			elif mat_info['phi'] < 0 or mat_info['phi'] > 45:
				return json_input_number*1000+98	# material phi value error 

	# density
	if isinstance(fb_0, (int,float)) == False:
		return json_input_number*1000+700	# equivalent basal resistance number error

	elif fb_0 < 0:
		return json_input_number*1000+701	# equivalent basal resistance value error

	##############################################################
	## gravity
	##############################################################
	if isinstance(g, (int,float)) == False:
		return json_input_number*1000+100	# gravity number error

	##############################################################
	## check for interpolation method
	##############################################################
	if isinstance(interp_method, str) == False:
		return json_input_number*1000+120		# interpolation method string error

	else:
		interp_method_list = interp_method.split(" ")

		if len(interp_method_list) == 1:
			if interp_method not in ['linear', 'cubic']:
				return json_input_number*1000+121		# interpolation method option error
		else:
			if interp_method_list[0] not in ['OK', 'UK']: 
				return json_input_number*1000+121		# interpolation method option error
			else:
				if interp_method_list[1] not in ['linear', 'power', 'gaussian', 'spherical', 'exponential']:
					return json_input_number*1000+122	# kriging interpolation semi-variogram option error
				
	##############################################################
	## check for entrainment method
	##############################################################
	if isinstance(entrainment_model, str) == False:
		return json_input_number*1000+130		# entrainment model string error
	else:
		if entrainment_model not in ['Hungr', 'Er']: 
			return json_input_number*1000+131		# entrainment model option error

	##############################################################
	## Es_theta_var
	##############################################################
	if isinstance(Es_theta_var, (int,float)) == False:
		return json_input_number*1000+150	# Es_theta_var number error

	elif Es_theta_var < 0 or Es_theta_var > 90:
		return json_input_number*1000+151	# Es_theta_var value error

	##############################################################
	## pathway algorithm - local cell size
	##############################################################
	if isinstance(cell_size, (int,float)) == False:
		return json_input_number*1000+162	# local cell size number error

	##############################################################
	## cpu multiprocessing number
	##############################################################
	if not( isinstance(max_cpu_num, int) ):
		return json_input_number*1000+210		# cpu multiprocessing number error

	elif max_cpu_num < 1:
		return json_input_number*1000+211		# cpu multiprocessing value error

	##############################################################
	## check for optimal barrier location selection
	##############################################################

	if not ( isinstance(opt_total_loop, int) and isinstance(opt_iter_max, int) and isinstance(opt_find_iter_limit, int) ):
		return json_input_number*1000+300		# optimal barrier location loop number error

	if not(opt_barrier_num_limit == [None, None]): 
		if not( isinstance(opt_barrier_num_limit[0], int) and isinstance(opt_barrier_num_limit[1], int) ):
			return json_input_number*1000+301		# optimal barrier location limit number error

		if opt_barrier_num_limit[0] > opt_barrier_num_limit[1] or opt_barrier_num_limit[0] < 0 or opt_barrier_num_limit[1] < 0:
			return json_input_number*1000+302		# optimal barrier location limit value error

	if isinstance(opt_avoid_dfl, bool) == False:
		return json_input_number*1000+303		# optimal barrier location avoid DFL option error

	# for combined closed and open-type barrier location selection
	# optimization weighting factor
	if int(sum([isinstance(owco, (int,float)) for owco in opt_weight_combined])) != len(opt_weight_combined):
		return json_input_number*1000+320		# optimal combined barrier number error
	elif round(sum(opt_weight_combined)) != 1:
		return json_input_number*1000+321		# optimal combined barrier sum value error

	# open type barrier performance 
	if int(sum([isinstance(opp, (int,float)) for opp in open_performance])) != 2:
		return json_input_number*1000+330		# open type barrier number error
	elif open_performance[0] < 0 or open_performance[0] > 1:
		return json_input_number*1000+331		# open type barrier SR value error
	elif open_performance[1] < 0 or open_performance[1] > 1:
		return json_input_number*1000+332		# open type barrier TR value error

	# vulnerability index
	if not (isinstance(min_uV[0], (int,float)) and isinstance(min_uV[1], (int,float)) and isinstance(VI_crit, (int,float)) and isinstance(RC_bool, bool)):
		return json_input_number*1000+340		# vulnerability analysis number error
	elif min_uV[0] < 0 or min_uV[1] < 0:
		return json_input_number*1000+341		# vulnerability analysis min_uV value error
	elif VI_crit < 0 or VI_crit > 1:
		return json_input_number*1000+342		# vulnerability analysis VI_crit value error

	if isinstance(output_optimal_step, bool) == False:
		return json_input_number*1000+350		# output_optimal_step option error

	##############################################################
	## plotting
	##############################################################
	# check boolean type
	if isinstance(plot_map_2D, bool) == False:
		return json_input_number*1000+400		# figure option error

	if not ( isinstance(marker_size, (int,float)) and isinstance(line_width, (int,float)) and isinstance(layout_width, (int,float)) and isinstance(layout_height, (int,float)) ):
		return json_input_number*1000+440		# figure size error
	
	elif (marker_size <= 0) or (line_width <= 0) or (layout_width <= 0) or (layout_height <= 0):
		return json_input_number*1000+441		# figure size value error

	##############################################################
	## output
	##############################################################
	# decimal point number
	if isinstance(dp, int) == False:
		return json_input_number*1000+510		# decimal point number error

	elif dp < 0:
		return json_input_number*1000+511		# decimal point value error

	return 0	# no error in the json input file

# overall check
def check_json_input_v8_00(json_file_name):

	error_message_code_title_text = {
		0: ['no error', 'no error found'], 
		1: ['JSON file heading error', 'the input file heading has violated the JSON format'], 
		2: ['JSON file variable error', 'the required variable name has a typo or is unavailable'], 
		3: ['python script file naming error', 'the naming of python file has error. Please ensure the python file name follows this format: SPEC_debris_barrier_platform_v00_00.py where 00_00 is the version number'], 
		4: ['python script version error', 'the version number on the python script and the input json file should match'], 
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
		811: ['part_input filetype error', 'the initial particle data should be an csv file format'], 
		999: ['unknown error', 'unidentified error, please check input once more']
	}

	# load json file
	with open(json_file_name) as json_file:
		json_opt_input_all_data = json.load(json_file)

	## iterate through each json input data

	# iterate through each key
	for json_key_number, json_input_key in enumerate(json_opt_input_all_data.keys()):

		# iterate through each input data in a given key
		for json_data_number, json_input_dict in enumerate(json_opt_input_all_data[json_input_key]): 

			# input data number
			json_input_number = (json_key_number+1)*100 + (json_data_number+1)

			if json_input_key == "SPEC-debris-closed":
				check_output = check_SPEC_debris_closed_json_input_v11_00(json_input_number, json_file_name, json_input_dict)

			elif json_input_key == "SPEC-debris-combined":
				check_output = check_SPEC_debris_combined_json_input_v11_00(json_input_number, json_file_name, json_input_dict)

			elif json_input_key == "SPEC-debris":
				check_output = check_SPEC_debris_json_input_v11_00(json_input_number, json_file_name, json_input_dict, wall_keyword=False)

			elif json_input_key == "SPEC-debris-wall":
				check_output = check_SPEC_debris_json_input_v11_00(json_input_number, json_file_name, json_input_dict, wall_keyword=True)

			elif json_input_key == "closed":
				check_output = check_closed_json_input_v8_00(json_input_number, json_file_name, json_input_dict)

			elif json_input_key == "combined":
				check_output = check_combined_json_input_v8_00(json_input_number, json_file_name, json_input_dict)

			else:
				check_output = json_input_number*1000+1  # JSON file heading error

			if check_output != 0:
				return (json_input_number, int(check_output%1000), error_message_code_title_text[int(check_output%1000)][0], error_message_code_title_text[int(check_output%1000)][1])


	return 0  	# return 0 if no error found anywhere in the JSON files

###########################################################################
## main function
###########################################################################
# class from imported csv files
# 'cID,t,s,x,y,z,u,h,V,D,P,A,Fr,CCHa,merged'
class cluster_class:
	def __init__(self, list_data):
		self.clusterID = list_data[0]
		self.time = list_data[1]
		self.sc = list_data[2]
		self.xc = list_data[3]
		self.yc = list_data[4]
		self.zc = list_data[5]
		self.uc = list_data[6]
		self.hc = list_data[7]
		self.Vc = list_data[8]
		self.Dc = list_data[9]
		self.Pc = list_data[10]
		self.area = list_data[11]
		self.Fr = list_data[12]
		self.concave_hull_alpha = list_data[13]
		self.merged = list_data[14]
		self.dfl_type = 0
		self.dfl_id = None

# only optimal closed-type barrier selection
def opt_closed_location_v8_00(json_file_name, json_input_dict):
	'''
	perform only optimal closed-type barrier location selction
	based on previously simulated debris-flow simulation
	'''

	## extract data from json input dictionary
	exportName = json_input_dict["project name"]
	folder_path = json_input_dict['folder path']
	flowpath_file_name = json_input_dict['flowpath file name']
	
	road_xy_list = json_input_dict['road xy']
	
	opt_total_loop = json_input_dict["optimal barrier location selection option"]["optimal total loop"]
	opt_avoid_dfl = json_input_dict["optimal barrier location selection option"]["optimal avoid dfl"]
	opt_iter_max = json_input_dict["optimal barrier location selection option"]["optimal iteration max"]
	opt_find_iter_limit = json_input_dict["optimal barrier location selection option"]["optimal find iteration limit"]
	opt_barrier_num_limit = [json_input_dict["optimal barrier location selection option"]["optimal barrier num min"],
								json_input_dict["optimal barrier location selection option"]["optimal barrier num max"]]
	
	opt_weight_close = [
		json_input_dict["closed barrier optimization criteria"]["volume(V)"],
		json_input_dict["closed barrier optimization criteria"]["pressure(P)"],
		json_input_dict["closed barrier optimization criteria"]["distance_from_road(D)"],
		json_input_dict["closed barrier optimization criteria"]["closed_barrier_number(N)"]
	]
	
	output_optimal_step = json_input_dict['output optimal step']
	
	plot_map_2D = json_input_dict["static plot option"]["plot map 2D"]

	marker_size = json_input_dict['marker size']
	line_width = json_input_dict['line width']
	layout_width = json_input_dict['layout width']
	layout_height = json_input_dict['layout height']
	
	dp = json_input_dict['decimal points']

	cluster_list_file_name = json_input_dict['SPEC_debris analysis cluster data']

	# folder path
	if folder_path == None:  # current multiple json file folder path == each simulation folder path
		folder_path = os.path.dirname(json_file_name)#+'/'
	else:
		folder_path = folder_path#+'/'

	## import json into class object		
	cluster_list_final = []
	for cluster_file_name in cluster_list_file_name:
	 
		# cluster_cID_array = np.loadtxt(folder_path+cluster_file_name, delimiter=',', skiprows=1, comments='')
		cluster_cID_array = np.loadtxt(folder_path+cluster_file_name, delimiter=',', skiprows=1)
		cluster_cID = cluster_cID_array.tolist()
		
		cluster_list_cID = []
		for cluster_data_list_i in cluster_cID:
			cluster_i = cluster_class(cluster_data_list_i)
			cluster_list_cID.append(cluster_i)
		
		cluster_list_final.append(deepcopy(cluster_list_cID))
		del cluster_list_cID

	## find optimal closed type barrier location
	final_optimal_dfl_barrier, best_min_cost, final_opt_dfl = iterate_find_optimal_closed_barrier_v2_0(folder_path, flowpath_file_name, cluster_list_final, road_xy_list, total_loop=opt_total_loop, opt_weight=opt_weight_close, avoid_dfl=opt_avoid_dfl, iteration_limit=opt_iter_max, find_iter_limit=opt_find_iter_limit, barrier_num_limit=opt_barrier_num_limit, output_step=output_optimal_step, export_step=output_optimal_step, plot_network=plot_map_2D, exportName=exportName, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height, dp=dp)

	## adjust final_opt_dfl cluster class
	final_opt_dfl_json = {}
	for DFLi, data_i in final_opt_dfl.items():
		temp_list = deepcopy(data_i)
		
		# convert class to dictionary
		temp_class_list = []
		for cluster_ii in data_i[1]:
			cluster_ii_dict = deepcopy(cluster_ii.__dict__) # export class into dictionary
			
			# remove predecessor, particles and boundary polygon (class type)
			if 'predecessor' in cluster_ii_dict:
				del cluster_ii_dict['predecessor']
			if 'particles' in cluster_ii_dict:		
				del cluster_ii_dict['particles']
			if 'boundary_polygon' in cluster_ii_dict:	
				del cluster_ii_dict['boundary_polygon']
			if 'boundary_pt_xy' in cluster_ii_dict:	
				del cluster_ii_dict['boundary_pt_xy']

			temp_class_list.append(deepcopy(cluster_ii_dict))
		
		temp_list[1] = deepcopy(temp_class_list)
  
		# create class for writing optimal dfl
		temp_class_result = {
			"cluster ID": temp_list[0],
			"cluster DFL parameters": temp_list[1],
			"optimization criteria": {
				"volume(V)": temp_list[2][0], 
				"pressure(P)": temp_list[2][1],
				"distance_from_road(D)": temp_list[2][2]
			},
			"normalized optimization criteria": {
				"normalized volume(V)": temp_list[3][0], 
				"normalized pressure(P)": temp_list[3][1], 
				"normalized distance_from_road(D)": temp_list[3][2]
			}
		}

		# final_opt_dfl_json[DFLi] = deepcopy(temp_list)
		final_opt_dfl_json[DFLi] = deepcopy(temp_class_result)
		del temp_class_result

	# generate output json file
	now = datetime.now()
	current_output_date_time = now.strftime("%Y/%m/%d %H:%M:%S")

	opt_closed_output = {
		# heading info
		'time-date': current_output_date_time,
		'project name': exportName,  # project name / export name
		'folder path': folder_path,	# project folder path

		# DEM data
		"flowpath file name": flowpath_file_name, 

		# road XY-coordinates
		'road xy': str(road_xy_list),

		# optimal barrier cost function weight list
		"optimal barrier location selection option": json_input_dict["optimal barrier location selection option"], 
		"closed barrier optimization criteria": json_input_dict["closed barrier optimization criteria"],
  
		# optimal closed-type barrier location selection
		"optimal barrier location selection summary": {
			"optimal closed-type barrier locations": final_optimal_dfl_barrier,
			"optimal open-type barrier locations": None,
			"optimal cost": best_min_cost,
			"optimal DFL data": final_opt_dfl_json,
			"terminus DFL data": None
		}
	}

	# create an output json file
	with open(folder_path+exportName+' - opt_closed_results.json', 'w') as fp:
		json.dump(opt_closed_output, fp, indent=4)

	return opt_closed_output

# only optimal combined open- and closed-type barrier selection
def opt_combined_location_v8_00(json_file_name, json_input_dict):
	'''
	perform only optimal open- and closed-type barrier location selction
	based on previously simulated debris-flow simulation
	'''

	## extract data from json file	
	exportName = json_input_dict["project name"]
	folder_path = json_input_dict['folder path']
	flowpath_file_name = json_input_dict['flowpath file name']

	road_xy_list = json_input_dict['road xy']

	material = {}
	for mat_key, mat_info in json_input_dict['material'].items():
		material[int(mat_key)] = mat_info

	MAT = np.array(json_input_dict['material id array'])
	DEM_no_wall = np.array(json_input_dict['flowpath z array'])
	gridUniqueX = json_input_dict['grid x array']
	gridUniqueY = json_input_dict['grid y array']
	deltaX = json_input_dict['grid delta_x']
	deltaY = json_input_dict['grid delta_y']
	fb_0 = json_input_dict['overall_fb_0']

	opt_total_loop = json_input_dict["optimal barrier location selection option"]["optimal total loop"]
	opt_avoid_dfl = json_input_dict["optimal barrier location selection option"]["optimal avoid dfl"]
	opt_iter_max = json_input_dict["optimal barrier location selection option"]["optimal iteration max"]
	opt_find_iter_limit = json_input_dict["optimal barrier location selection option"]["optimal find iteration limit"]
	opt_barrier_num_limit = [json_input_dict["optimal barrier location selection option"]["optimal barrier num min"],
								json_input_dict["optimal barrier location selection option"]["optimal barrier num max"]]

	open_performance = [
		json_input_dict["open barrier performance"]["speed_ratio(SR)"],
		json_input_dict["open barrier performance"]["trap_ratio(TR)"]
	]
	opt_weight_combined = [
		json_input_dict["combined barrier optimization criteria"]["volume(V)"],
		json_input_dict["combined barrier optimization criteria"]["pressure(P)"],
		json_input_dict["combined barrier optimization criteria"]["distance_from_road(D)"],
		json_input_dict["combined barrier optimization criteria"]["closed_barrier_number(NC)"],
		json_input_dict["combined barrier optimization criteria"]["opened_barrier_number(NO)"]
	]

	cell_size = max(json_input_dict['local cell sizes'])
	g = json_input_dict["gravitational acceleration"] 
	entrainment_model = json_input_dict['entrainment model'] 
	Es_theta_var = json_input_dict['free fall angle variation'] 
	interp_method = json_input_dict['interpolation method'] 
	min_uV = [0.0, 0.0] 
	VI_crit = json_input_dict['critical vulnerability index (VI)'] 
	RC_bool = json_input_dict['reinforced concrete (RC) wall'] 

	output_optimal_step = json_input_dict['output optimal step']
	
	plot_map_2D = json_input_dict["static plot option"]["plot map 2D"]
			
	marker_size = json_input_dict['marker size']
	line_width = json_input_dict['line width']
	layout_width = json_input_dict['layout width']
	layout_height = json_input_dict['layout height']
	
	max_cpu_num = json_input_dict["max cpu num"]

	dp = json_input_dict['decimal points']

	cluster_list_file_name = json_input_dict['SPEC_debris analysis cluster data']

	# folder path
	if folder_path == None:  # current multiple json file folder path == each simulation folder path
		folder_path = os.path.dirname(json_file_name)+'/'
	else:
		folder_path = folder_path+'/'

	# import json into class object
	cluster_list_final = []
	for cluster_file_name in cluster_list_file_name:
	 
		# cluster_cID_array = np.loadtxt(folder_path+cluster_file_name, delimiter=',', skiprows=1, comments='')
		cluster_cID_array = np.loadtxt(folder_path+cluster_file_name, delimiter=',', skiprows=1)
		cluster_cID = cluster_cID_array.tolist()
		
		cluster_list_cID = []
		for cluster_data_list_i in cluster_cID:
			cluster_i = cluster_class(cluster_data_list_i)
			cluster_list_cID.append(cluster_i)
		
		cluster_list_final.append(deepcopy(cluster_list_cID))
		del cluster_list_cID
		
	## find optimal open-type barrier locations
	optimal_dfl_combined_barrier_set, combined_barrier_cost, combined_barrier_dfl_data, terminus_dfl_data = iterate_find_optimal_combined_barrier_v3_0(folder_path, flowpath_file_name, cluster_list_final, road_xy_list, material, MAT, DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, fb_0, alpha=1.5, total_loop=opt_total_loop, open_performance=open_performance, opt_weight=opt_weight_combined, avoid_dfl=opt_avoid_dfl, iteration_limit=opt_iter_max, find_iter_limit=opt_find_iter_limit, cell_size=cell_size, g=g, entrainment_model=entrainment_model, Es_theta_var=Es_theta_var, interp_method=interp_method, min_uV=min_uV, VI_crit=VI_crit, RC_bool=RC_bool, barrier_num_limit=opt_barrier_num_limit, output_step=output_optimal_step, plot_network=plot_map_2D, exportName=exportName, max_cpu_num=max_cpu_num, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height, dp=dp)
 
 
	# combined_barrier_dfl_data = {} # key = dfl, value = [barrier_performance, [cID_list], [V, P, D] [normalized V, P, D]
	combined_barrier_dfl_data_mod = {}
	for DFLi, data_i in combined_barrier_dfl_data.items():
		temp_list = deepcopy(data_i)
  
		# create class for writing optimal dfl
		temp_class_result = {
			"barrier performance": temp_list[0],
			"cluser ID": temp_list[1],
			"optimization criteria": {
				"volume(V)": temp_list[2][0], 
				"pressure(P)": temp_list[2][1],
				"distance_from_road(D)": temp_list[2][2]
			},
			"normalized optimization criteria": {
				"normalized volume(V)": temp_list[3][0], 
				"normalized pressure(P)": temp_list[3][1], 
				"normalized distance_from_road(D)": temp_list[3][2]
			}
		}

		combined_barrier_dfl_data_mod[DFLi] = deepcopy(temp_class_result)
		del temp_class_result, temp_list
  
	# dfl_terminus_VI = {}  # {terminus_dfl : [velocity, VI]}
	dfl_terminus_VI_mod = {} 
	for tDFLi, data_i in terminus_dfl_data.items():
		temp_list = deepcopy(data_i)
  
		# create class for writing optimal dfl
		temp_class_result = {
			"velocity at terminus": temp_list[0],
			"vulnerability index (VI) from velocity": temp_list[1]
		}

		dfl_terminus_VI_mod[tDFLi] = deepcopy(temp_class_result)
		del temp_class_result, temp_list

	now = datetime.now()
	current_output_date_time = now.strftime("%Y/%m/%d %H:%M:%S")

	opt_combined_output = {
		# heading info
		'time-date': current_output_date_time,
		'project name': exportName,  # project name / export name
		'folder path': folder_path,	# project folder path

		# DEM data
		"flowpath file name": flowpath_file_name, 

		# road XY-coordinates
		'road xy': str(road_xy_list),

		# optimization search options
		"optimal barrier location selection option": json_input_dict["optimal barrier location selection option"],
		"combined barrier optimization criteria": json_input_dict["combined barrier optimization criteria"],
		"open barrier performance": json_input_dict["open barrier performance"],
		"critical vulnerability index (VI)":json_input_dict["critical vulnerability index (VI)"],
		"reinforced concrete (RC) wall": json_input_dict["reinforced concrete (RC) wall"], 

		# optimal combined (closed and open) barrier 
		"optimal barrier location selection summary": {
			"optimal closed-type barrier locations": optimal_dfl_combined_barrier_set[0],
			"optimal open-type barrier locations": optimal_dfl_combined_barrier_set[1],
			"optimal cost": combined_barrier_cost,
			"optimal DFL data": combined_barrier_dfl_data_mod,
			"terminus DFL data": dfl_terminus_VI_mod
		}
	}

	# create an output json file
	with open(folder_path+exportName+' - opt_combined_results.json', 'w') as fp:
		json.dump(opt_combined_output, fp, indent=4)

	return opt_combined_output

# SPEC-debris simulation and optimal closed-type barrier selection
def SPEC_debris_closed_v11_50(json_file_name, json_input_dict):

	"""
	SPEC-debris incoporates multiprocessing v1.00

	Creator: Enok Cheon
	Date: 2022-04-11
	Language: Python3
	License: MIT

	perform SPEC-debris simulation for the following purpose:
		1) debris-flow propagation (with and without barrier)
		2) barrier performance against debris-flow collision

	Entrainment model options:
		'Hungr': E(s) = exp(Es*ds)
		'Er': E(s) = Es*ds

	Explanation of the json input file format for SPEC-debris are described in the user manual
	"""

	##################################################################################################################
	## read json file and extract input file data
	##################################################################################################################

	try: # exception for json input file error

		###########################################################
		## Name and folder information
		###########################################################
		# project name / export name
		exportName = json_input_dict["project name"] 

		# project folder path
		folder_path = json_input_dict["folder path"]
		if folder_path == None:  # current multiple json file folder path == each simulation folder path
			folder_path = os.path.dirname(json_file_name)+'/'
		else:
			folder_path = folder_path+'/'

		###########################################################
		## map data - DEM, road, wall/building
		###########################################################
		# DEM data
		flowpath_file_name = json_input_dict["flowpath file name"]
		source_file_name = json_input_dict["source file name"]
		material_file_name = json_input_dict["material file name"]

		# road and goal XY-coordinates
		road_xy_list = [tuple(xy_pt) for xy_pt in json_input_dict["road xy"]]

		###########################################################
		## SPEC-debris analysis set-up and material properties
		###########################################################
		# time step parameters
		t_step = json_input_dict["time step interval"]
		t_max = json_input_dict["maximum simulation time"]

		# particle number
		if json_input_dict["particle number per cell"] is None:
			part_num_per_cell = 1
		else:
			try:
				part_num_per_cell = int(json_input_dict["particle number per cell"])
			except:
				part_num_per_cell = 1

		# initial setup of debris-flow - initial velocity
		initial_velocity = (json_input_dict["initial u_x"], json_input_dict["initial u_y"])

		# material
		material = {}
		for mat_key, mat_info in json_input_dict["material"].items():
			material[int(mat_key)] = mat_info

		# load - gravity
		g = json_input_dict["gravitational acceleration"]

		# interaction - coefficient of restitution COR
		COR = (json_input_dict["coefficient of restitution (COR)"]["particle with particle"], 
				json_input_dict["coefficient of restitution (COR)"]["particle with wall"])
		# COR = (1.0, json_input_dict["coefficient of restitution (COR)"]["particle with wall"]) 

		# perform SPH interpolation at time step = 0
		initial_SPH = json_input_dict["initial SPH"] 

		###########################################################
		## Algorithm options 
		###########################################################
		# interpolation method
		interp_method = json_input_dict["interpolation method"]

		# Entrainment model
		entrainment_model = json_input_dict["entrainment model"]

		# cluster boundary algorithm
		cluster_boundary = json_input_dict["cluster boundary method"]	

		###########################################################
		## Control parameters 
		###########################################################
		# no entrainment when climbing wall or free-falling
		Es_theta_var = json_input_dict["free fall angle variation"]

		# simulation coefficients and constants
		cell_size = tuple(json_input_dict["local cell sizes"]) 		

		# SPH smoothing length computation
		l_dp_min = json_input_dict["l_dp_min"]

		# cluster boundary control parameters for Concave-Hull
		concave_hull_parameter_dict = json_input_dict["concave hull algorithm parameter"]

		# mutliple cluster merge
		merge_overlap_ratio = json_input_dict["merge overlap ratio"]

		# multiprocessing
		max_cpu_num = json_input_dict["max cpu num"]

		###########################################################
		## optimal barrier location options - closed-type only
		###########################################################

		# selection of optimal barrier parameters 
		opt_total_loop = json_input_dict["optimal barrier location selection option"]["optimal total loop"]
		opt_avoid_dfl = json_input_dict["optimal barrier location selection option"]["optimal avoid dfl"]
		opt_iter_max = json_input_dict["optimal barrier location selection option"]["optimal iteration max"]
		opt_find_iter_limit = json_input_dict["optimal barrier location selection option"]["optimal find iteration limit"]
		opt_barrier_num_limit = [json_input_dict["optimal barrier location selection option"]["optimal barrier num min"],
									json_input_dict["optimal barrier location selection option"]["optimal barrier num max"]]

		# track cost per epoch
		output_optimal_step = json_input_dict["output optimal step"]

		# only closed-type barrier
		opt_weight_close = [
			json_input_dict["closed barrier optimization criteria"]["volume(V)"],
			json_input_dict["closed barrier optimization criteria"]["pressure(P)"],
			json_input_dict["closed barrier optimization criteria"]["distance_from_road(D)"],
			json_input_dict["closed barrier optimization criteria"]["closed_barrier_number(N)"]
		]
		
		###########################################################
		## plotting options 
		###########################################################
		
		# plotting options
		plot_map_2D = json_input_dict["static plot option"]["plot map 2D"]
		plot_map_3D = json_input_dict["static plot option"]["plot map 3D"]
				
		# animation options
		plot_animation_2D = json_input_dict["animation option"]["plot animation 2D"]
		plot_animation_3D = json_input_dict["animation option"]["plot animation 3D"]
		plot_animation_2D_boundary = json_input_dict["animation option"]["plot animation 2D boundary"]

		# open plot after creating
		open_plot = json_input_dict["open plot"]

		# 3D plot option
		plot_3D_z_offset = json_input_dict["static plot option"]["plot 3D z offset"]

		# animation options
		animation = [
			json_input_dict["animation option"]["frame duration"], 
			json_input_dict["animation option"]["frame transition"], 
			json_input_dict["animation option"]["contour elevation interval"]
		]
		animation_3D_z_offset = json_input_dict["animation option"]["animation 3D z offset"]

		# plot maximum legend
		plot_2D_max_limits = [
			json_input_dict["max parameter legend"]["cluster max velocity(u)"],
			json_input_dict["max parameter legend"]["cluster max depth(h)"],
			json_input_dict["max parameter legend"]["cluster max volume(V)"],
			json_input_dict["max parameter legend"]["cluster max distance_from_road(D)"],
			json_input_dict["max parameter legend"]["cluster max pressure(P)"],
			json_input_dict["max parameter legend"]["particle max velocity(u)"],
			json_input_dict["max parameter legend"]["particle max depth(h)"],
			json_input_dict["max parameter legend"]["particle max volume(V)"],
			json_input_dict["max parameter legend"]["particle max pressure(P)"]
		]
		plot_3D_max_limits = deepcopy(plot_2D_max_limits)

		# marker, line, and plot size options
		marker_size = json_input_dict["marker size"]
		line_width = json_input_dict["line width"]
		layout_width = json_input_dict["layout width"]
		layout_height = json_input_dict["layout height"]

		###########################################################
		## output options 
		###########################################################
		# csv data export for SPEC-debris analysis
		csv_output = json_input_dict["csv output"]

		# data decimal places
		dp = json_input_dict["decimal points"]

	except (KeyError, NameError):
		print()
		print('json variable(s) name or value are missing or incorrect; please check input json file')
		print()
		return None

	##################################################################################################################
	## import data (las or grd or asc -> csv) for flowpath_file_name, source_file_name, and material_file_name
	##################################################################################################################

	try: # input file error

		##############################################################
		## flowpath_file_name
		##############################################################
		# find file type [csv, las, grd]
		flowpath_file_name_list = flowpath_file_name.split('.')
		flowpath_file_name_type = flowpath_file_name_list[-1]

		if flowpath_file_name_type == 'csv':
			pass
		
		elif flowpath_file_name_type in ['las', 'grd', 'asc']:

			# find the file name only
			flowpath_file_name_only_list = flowpath_file_name.split("."+flowpath_file_name_type)
			flowpath_file_name_only_list2 = [txt if n==0 or n==len(flowpath_file_name_only_list)-1 else flowpath_file_name_only_list2[n-1]+"."+flowpath_file_name_type+txt for n,txt in enumerate(flowpath_file_name_only_list)]
			flowpath_file_name_only = flowpath_file_name_only_list2[-2]

			# create xyz data and convert to csv data
			if flowpath_file_name_type == 'las':
				flowpath_database = las2xyz(folder_path+flowpath_file_name_only, outFileName=folder_path+flowpath_file_name_only, outFileFormat='csv', saveOutputFile=True)

			elif flowpath_file_name_type == 'grd':
				flowpath_database = grd2xyz(folder_path+flowpath_file_name_only, headDataOutput=False, outFileName=folder_path+flowpath_file_name_only, saveOutputFile=True)
		
			elif flowpath_file_name_type == 'asc':
				flowpath_database = asc2xyz(folder_path+flowpath_file_name_only, outFileName=folder_path+flowpath_file_name_only, saveOutputFile=True)

			flowpath_file_name = folder_path+flowpath_file_name_only+'.csv'

		else:
			raise RuntimeError

		##############################################################
		## source_file_name
		##############################################################
		# find file type [csv, las, grd]
		source_file_name_list = source_file_name.split('.')
		source_file_name_type = source_file_name_list[-1]

		if source_file_name_type == 'csv':
			pass
		
		elif source_file_name_type in ['las', 'grd', 'asc']:

			# find the file name only
			source_file_name_only_list = source_file_name.split("."+source_file_name_type)
			source_file_name_only_list2 = [txt if n==0 or n==len(source_file_name_only_list)-1 else source_file_name_only_list2[n-1]+"."+source_file_name_type+txt for n,txt in enumerate(source_file_name_only_list)]
			source_file_name_only = source_file_name_only_list2[-2]

			# create xyz data and convert to csv data
			if source_file_name_type == 'las':
				source_database = las2xyz(folder_path+source_file_name_only, outFileName=folder_path+source_file_name_only, outFileFormat='csv', saveOutputFile=True)

			elif source_file_name_type == 'grd':
				source_database = grd2xyz(folder_path+source_file_name_only, headDataOutput=False, outFileName=folder_path+source_file_name_only, saveOutputFile=True)

			elif source_file_name_type == 'asc':
				source_database = asc2xyz(folder_path+source_file_name_only, outFileName=folder_path+source_file_name_only, saveOutputFile=True)

			source_file_name = folder_path+source_file_name_only+'.csv'

		else:
			raise RuntimeError

		##############################################################
		## material_file_name
		##############################################################
		# import material file if number of number of material is higher than 1
		if material_file_name != None and len(material.keys()) > 1:
			# find file type [csv, las, grd]
			material_file_name_list = material_file_name.split('.')
			material_file_name_type = material_file_name_list[-1]

			if material_file_name_type == 'csv':
				pass
			
			elif material_file_name_type in ['las', 'grd', 'asc']:

				# find the file name only
				material_file_name_only_list = source_file_name.split("."+material_file_name_type)
				material_file_name_only_list2 = [txt if n==0 or n==len(material_file_name_only_list)-1 else material_file_name_only_list2[n-1]+"."+material_file_name_type+txt for n,txt in enumerate(material_file_name_only_list)]
				material_file_name_only = material_file_name_only_list2[-2]

				# create xyz data and convert to csv data
				if material_file_name_type == 'las':
					material_database = las2xyz(folder_path+material_file_name_only, outFileName=folder_path+material_file_name_only, outFileFormat='csv', saveOutputFile=True)

				elif material_file_name_type == 'grd':
					material_database = grd2xyz(folder_path+material_file_name_only, headDataOutput=False, outFileName=folder_path+material_file_name_only, saveOutputFile=True)

				elif material_file_name_type == 'asc':
					material_database = asc2xyz(folder_path+material_file_name_only, outFileName=folder_path+material_file_name_only, saveOutputFile=True)
			
				material_file_name = folder_path+material_file_name_only+'.csv'

			else:
				raise RuntimeError

		# import data issue
		elif material_file_name == None and len(material.keys()) > 1:
			raise KeyError

		elif len(material.keys()) == 1:
			pass

	except RuntimeError:
		print()
		print('please input csv or las (LiDAR) or grd (Surfer) for the flowpath/source/material file')
		print()
		return None
	except KeyError:
		print()
		print('please specify the material_file_name as number of material types > 1')
		print()
		return None

	##################################################################################################################
	## extract flowpath (elevation) and material data
	##################################################################################################################

	##############################################################
	## flowpath 
	##############################################################
	DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True) 
 
	##############################################################
	## eroding depth
	##############################################################
	ERODE = np.zeros(DEM_no_wall.shape)
 
	##############################################################
	## material
	##############################################################
	if material_file_name != None:
		MAT = xyz2mesh(folder_path+material_file_name, exportAll=False) 
	else:
		MAT = np.ones(DEM_no_wall.shape)

	##################################################################################################################
	## stable time step approach - without analyzing barrier performance
	##################################################################################################################

	# error message
	if road_xy_list == None:
		assert False, 'specify the road_xy_list to compute distance_from_road (D) factor'

	# parameters/variables related to wall
	wall_info = None
	wall_dict = None
	flowpath_file_name_with_wall = None
	DEM_with_wall = None
	wall_bound_region = None
	wall_perform_region = None
	wall_performance = None

	##############################################################
	## generate particle data at time step = 0
	##############################################################
	# generate particle and cluster data			
	cluster_list0, clusterID_flow_list0, max_cID_num, all_part_list0, part_num, part_radius = time0_setup_t_v12_0(folder_path+source_file_name, folder_path+flowpath_file_name, part_num_per_cell, cell_size[0], DEM_no_wall, MAT, material, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, road_xy_list, initial_velocity, g, wall_info, cluster_boundary, concave_hull_parameter_dict, l_dp_min, t_step, DP=dp, particle_only=False, initial_SPH=initial_SPH)

	# generate overall output summary data for time = 0
	overall_output_raw = np.array([[part_i.time, part_i.si, part_i.x, part_i.y, part_i.z, part_i.ui, part_i.hi, part_i.Vi] for part_i in all_part_list0[0]])
	overall_output_mean = np.mean(overall_output_raw[:,:-1], axis=0).tolist()
	overall_output_sum = np.sum(overall_output_raw[:,-1], axis=0).tolist()
	overall_output_summary0 = [overall_output_mean+[overall_output_sum]]

	print()
	print('completed setting initial simulation setting:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## SPEC-debris simulation for each weight factor sets
	##############################################################
	
	## SPEC-debris for the selected weight coefficients
	# cluster_list_final, overall_output_summary, all_part_list, t_track = SPEC_debris_c_t_nb_MP_v9_0(DEM_no_wall, ERODE, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, cluster_list0, clusterID_flow_list0, max_cID_num, all_part_list0, overall_output_summary0, road_xy_list, material, COR, Es_theta_var, cell_size, part_radius, l_dp_min, g, merge_overlap_ratio, cluster_boundary, concave_hull_parameter_dict, interp_method, entrainment_model, t_step, t_max, max_cpu_num)
	cluster_list_final, overall_output_summary, all_part_list, t_track = SPEC_debris_c_t_nb_MP_v9_0(DEM_no_wall, ERODE, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, cluster_list0, clusterID_flow_list0, max_cID_num, all_part_list0, overall_output_summary0, road_xy_list, material, COR, Es_theta_var, cell_size, part_radius, l_dp_min, g, merge_overlap_ratio, cluster_boundary, concave_hull_parameter_dict, interp_method, entrainment_model, t_step, t_max, max_cpu_num, csv_output, exportName, folder_path, DP=dp)

	print()
	print('completed SPEC-debris simulation:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## simulation export
	##############################################################
	## sort data and export into csv file
	# store particles all data
	all_part_data = []  # 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'
	# everything_part_data = []
	max_recorded_part_uhP = [0, 0, 0]
	volume_part_track = [0, 0] # initial, final
	latest_t_stored = 0
	for time_step, part_list_it in enumerate(all_part_list):

		all_part_data_temp = []
		# everything_part_data_t = []
		for part_i in part_list_it:
			# store particle data
			all_part_data_temp.append(list(part_i.return_all_param()))
			# everything_part_data_t.append(list(part_i.return_everything()))

			## SPEC analysis results
			# initial particles - total volume
			if time_step == 0:
				volume_part_track[0] += part_i.Vi
			# final particles - total volume
			elif time_step >= len(all_part_list)-1:
				volume_part_track[1] += part_i.Vi

			# maximum recorded particle velocity
			if max_recorded_part_uhP[0] < part_i.ui:
				max_recorded_part_uhP[0] = part_i.ui

			# maximum recorded particle depth
			if max_recorded_part_uhP[1] < part_i.hi:
				max_recorded_part_uhP[1] = part_i.hi

			# maximum recorded particle impact pressure
			if max_recorded_part_uhP[2] < part_i.Pi:
				max_recorded_part_uhP[2] = part_i.Pi

		all_part_data.append(all_part_data_temp)
		# everything_part_data.append(everything_part_data_t)

	# store each cluster data
	cluster_summary_dict = {}	# key = cluster_id, value = [max(S), max(u), max(h), max(P), V_t0, V_final]
	for cID in range(max_cID_num+1):
		cluster_summary_dict[str(cID)] = {
			"cluster volume at initial": 0, 
			"cluster volume at final": 0, 
			"total travel distance": 0, 
			"max recorded cluster velocity": 0, 
			"max recorded cluster depth": 0, 
			"max recorded cluster pressure": 0
		}

	all_cluster_data = []  # 'cID,t,s,x,y,z,u,h,V,D,P,area,Fr,CCH_alpha,merged'
	all_cluster_boundary_poly_dict = {}	# contains external boundary through shapely polygon
	for cluster_it in cluster_list_final:

		all_cluster_data_temp = []
		# all_cluster_boundary_poly_temp = []
		latest_t_stored = 0
		for time_step, cluster_i in enumerate(cluster_it):

			all_cluster_data_temp.append(list(cluster_i.return_all()))

			## shapely polygon - boundary
			# all_cluster_boundary_poly_temp.append(cluster_i.boundary_polygon)

			if int(cluster_i.clusterID) not in all_cluster_boundary_poly_dict.keys():
				all_cluster_boundary_poly_dict[int(cluster_i.clusterID)] = {
					time_step: cluster_i.boundary_polygon
				}

			else:
				temp_bound_dict = deepcopy(all_cluster_boundary_poly_dict[int(cluster_i.clusterID)])
				temp_bound_dict[time_step] = cluster_i.boundary_polygon
				all_cluster_boundary_poly_dict[int(cluster_i.clusterID)] = deepcopy(temp_bound_dict)
				del temp_bound_dict

			## SPEC analysis results
			# initial total volume
			if time_step == 0:
				cluster_summary_dict[str(cluster_i.clusterID)]["cluster volume at initial"] = cluster_i.Vc

			# final total volume and max travel distance
			elif time_step == len(cluster_it)-1:
				cluster_summary_dict[str(cluster_i.clusterID)]["cluster volume at final"] = cluster_i.Vc
				cluster_summary_dict[str(cluster_i.clusterID)]["total travel distance"] = cluster_i.sc

			# maximum recorded particle velocity
			if cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster velocity"] < cluster_i.uc:
				cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster velocity"] = cluster_i.uc

			# maximum recorded particle depth
			if cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster depth"] < cluster_i.hc:
				cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster depth"] = cluster_i.hc

			# maximum recorded particle impact pressure
			if cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster pressure"] < cluster_i.Pc:
				cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster pressure"] = cluster_i.Pc

		all_cluster_data.append(deepcopy(all_cluster_data_temp))
		# all_cluster_boundary_poly_list.append(deepcopy(all_cluster_boundary_poly_temp))					

	## clusters - each cluster data - must be exported
	header_cluster = 'cID,t,s,x,y,z,u,h,V,D,P,A,Fr,CCHa,merged'
	cluster_summary_name_list = []
	for cID, loop_cluster_list in enumerate(all_cluster_data):
		cluster_summary_name = exportName+'_cluster'+str(cID)+'.csv'
		cluster_summary_name_list.append(cluster_summary_name)
		np.savetxt(folder_path+cluster_summary_name, loop_cluster_list, fmt='%.4f', delimiter=',', comments='', header=header_cluster)

	# store overall data
	all_output_summary_data = []	# 't,s,x,y,z,u,h,V'
	latest_t_stored = 0
	for overall_list_it in overall_output_summary:
		all_output_summary_data.append(deepcopy(overall_list_it))

	if csv_output:

		if exportName == None:
			exportName = 'SPEC-debris'
		
		## overall output - overall all particle summary data
		header_output = 't,s,x,y,z,u,h,V'
		np.savetxt(folder_path+exportName+'_overall_output.csv', all_output_summary_data, fmt='%.4f', delimiter=',', comments='', header=header_output)
	
	print()
	print('completed tabulating simulation results:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## simulation plotting
	##############################################################

	## 2D - plot static maps
	if plot_map_2D:
		plot_SPEC_debris_map_v6_0(folder_path, flowpath_file_name, all_part_data, all_cluster_data, all_output_summary_data, road_xy_list, exportName, max_limits=plot_2D_max_limits, open_html=open_plot, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height, max_cpu_num=max_cpu_num)

		print()
		print('completed 2D plotting simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## 2D - plot animation
	if plot_animation_2D:
		plot_SPEC_debris_animation_2D_plotly_v6_0(folder_path, flowpath_file_name, all_part_data, all_cluster_data, all_cluster_boundary_poly_dict, exportName, wall_dict, plot_animation_2D_boundary, animation_duration=animation[0], animation_transition=animation[1], contour_diff=animation[2], max_limits=plot_2D_max_limits, open_html=open_plot, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

		print()
		print('completed 2D animating simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## 3D - plot static maps
	if plot_map_3D:
		plot_SPEC_debris_surface_v4_0(folder_path, flowpath_file_name, all_part_data, all_cluster_data, exportName, max_limits=plot_3D_max_limits, open_html=open_plot, z_offset=plot_3D_z_offset, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

		print()
		print('completed 3D plotting simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## 3D - plot animation	
	if plot_animation_3D:
		plot_SPEC_debris_animation_3D_plotly_v5_0(folder_path, flowpath_file_name, all_part_data, all_cluster_data, exportName, animation_duration=animation[0], animation_transition=animation[1], contour_diff=animation[2], max_limits=plot_3D_max_limits, open_html=open_plot, z_offset=animation_3D_z_offset, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

		print()
		print('completed 3D animating simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	print()
	print('completed plotting simulation results:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## optimal barrier location
	##############################################################
	
	# find optimal closed-type barrier location(s)
	final_optimal_dfl_barrier, best_min_cost, final_opt_dfl = iterate_find_optimal_closed_barrier_v2_0(folder_path, flowpath_file_name, cluster_list_final, road_xy_list, total_loop=opt_total_loop, opt_weight=opt_weight_close, avoid_dfl=opt_avoid_dfl, iteration_limit=opt_iter_max, find_iter_limit=opt_find_iter_limit, barrier_num_limit=opt_barrier_num_limit, output_step=output_optimal_step, export_step=output_optimal_step, plot_network=plot_map_2D, exportName=exportName, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height, dp=dp)

	print()
	print('completed searching for optimal closed-type barriers:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## adjust final_opt_dfl cluster class
	final_opt_dfl_json = {}
	for DFLi, data_i in final_opt_dfl.items():
		temp_list = deepcopy(data_i)
		
		# convert class to dictionary
		temp_class_list = []
		for cluster_ii in data_i[1]:
			cluster_ii_dict = deepcopy(cluster_ii.__dict__) # export class into dictionary
			
			# remove predecessor, particles and boundary polygon (class type)
			del cluster_ii_dict['predecessor']
			del cluster_ii_dict['particles']
			del cluster_ii_dict['boundary_polygon']
			del cluster_ii_dict['boundary_pt_xy']
			temp_class_list.append(deepcopy(cluster_ii_dict))
		
		temp_list[1] = deepcopy(temp_class_list)
  
		# create class for writing optimal dfl
		temp_class_result = {
			"cluster ID": temp_list[0],
			"cluster DFL parameters": temp_list[1],
			"optimization criteria": {
				"volume(V)": temp_list[2][0], 
				"pressure(P)": temp_list[2][1],
				"distance_from_road(D)": temp_list[2][2]
			},
			"normalized optimization criteria": {
				"normalized volume(V)": temp_list[3][0], 
				"normalized pressure(P)": temp_list[3][1], 
				"normalized distance_from_road(D)": temp_list[3][2]
			}
		}

		# final_opt_dfl_json[DFLi] = deepcopy(temp_list)
		final_opt_dfl_json[DFLi] = deepcopy(temp_class_result)
		del temp_class_result
	
	now = datetime.now()
	current_output_date_time = now.strftime("%Y/%m/%d %H:%M:%S")

	temp_output = {
		# heading info
		'time-date': current_output_date_time,
		'project name': json_input_dict["project name"],  # project name / export name
		'folder path': folder_path,	# project folder path

		# DEM data
		"flowpath file name": json_input_dict["flowpath file name"], 
		'flowpath file name with wall': flowpath_file_name_with_wall, 
		"source file name": json_input_dict["source file name"],
		"material file name": json_input_dict["material file name"],

		# road and goal XY-coordinates
		'road xy': str(json_input_dict["road xy"]),

		# initial setup of debris-flow - initial velocity and particle number
		'initial_velocity': {
			"initial u_x": json_input_dict["initial u_x"], 
			"initial u_y": json_input_dict["initial u_y"]
		},
		"particle number": part_num,

		'material': json_input_dict["material"], # material
		'gravitational acceleration': g,	# load - gravity
		"coefficient of restitution (COR)":{
			"particle with particle": COR[0],
			"particle with wall": COR[1]
		},

		# no entrainment when climbing wall or free-falling
		'free fall angle variation': Es_theta_var,

		# time or distance controlled analysis
		'local cell sizes': str(json_input_dict["local cell sizes"]),

		# interpolation method
		'interpolation method': json_input_dict["interpolation method"],

		# Entrainment model
		'entrainment model': json_input_dict["entrainment model"],

		# mutliple cluster merge
		'merge overlap ratio': json_input_dict["merge overlap ratio"],
		
		# cluster boundary
		'cluster boundary method': cluster_boundary,
		'concave hull algorithm parameter': None if cluster_boundary=="ConvexHull" else concave_hull_parameter_dict,

		## simulation for time-step and barrier performance
		# time step parameters
		"time step interval": t_step,
		"maximum simulation time": round(t_track, dp), 

		# SPEC analysis results 
		"particle summary": {
			"total initial volume": volume_part_track[0],
			"total final volume": volume_part_track[1],
			"max recorded particle velocity": max_recorded_part_uhP[0],
			"max recorded particle depth": max_recorded_part_uhP[1],
			"max recorded particle impact pressure": max_recorded_part_uhP[2]
		},
 
		# SPEC cluster analysis results 
		"cluster summary": cluster_summary_dict,
  
		# optimal search data
		"optimal barrier location selection option": json_input_dict["optimal barrier location selection option"], 
		"closed barrier optimization criteria": json_input_dict["closed barrier optimization criteria"],

		# optimal closed-type barrier location selection
		"optimal barrier location selection summary": {
			"optimal closed-type barrier locations": final_optimal_dfl_barrier,
			"optimal open-type barrier locations": None,
			"optimal cost": best_min_cost,
			"optimal DFL data": final_opt_dfl_json,
			"terminus DFL data": None
		}
	}

	# create an output json file
	with open(folder_path+exportName+'.json', 'w') as fp:
		json.dump(temp_output, fp, indent=4)

	print()
	print('completed creating summary JSON file:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))
 
	## export json files with all inputs so that only the optimal barrier location can be run

	opt_close_output = {
		'closed': [
			{
				"project name": exportName,
				'folder path': folder_path,
				'flowpath file name': json_input_dict["flowpath file name"], 
				'road xy': json_input_dict["road xy"],
				"optimal barrier location selection option":{
					"optimal total loop": opt_total_loop, 
					"optimal avoid dfl": opt_avoid_dfl, 
					"optimal iteration max": opt_iter_max, 
					"optimal find iteration limit": opt_find_iter_limit, 
					"optimal barrier num min": opt_barrier_num_limit[0], 
					"optimal barrier num max": opt_barrier_num_limit[1]
				}, 
				"closed barrier optimization criteria": {
					"volume(V)": opt_weight_close[0], 
					"pressure(P)": opt_weight_close[1],
					"distance_from_road(D)": opt_weight_close[2],
					"closed_barrier_number(N)": opt_weight_close[3]
				},
				'output optimal step': output_optimal_step,
				"static plot option": {
					"plot map 2D": plot_map_2D,
					"plot map 3D": plot_map_3D,
					"plot 3D z offset": plot_3D_z_offset
				},
				'marker size': marker_size,
				'line width': line_width,
				'layout width': layout_width,
				'layout height': layout_height,
				'decimal points': dp,
				'SPEC_debris analysis cluster data': cluster_summary_name_list
			}
		]
	}

	# create an output json file
	with open(folder_path+exportName+' - opt_closed_input.json', 'w') as fp:
		json.dump(opt_close_output, fp, indent=4)

	print()
	print('completed generating input JSON file for optimal closed-type barrier location search:*********************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	# if everything goes well - return code to signify all completed well
	return temp_output    # no error occurred - code 0 

# SPEC-debris simulation and combined open- and closed-type barrier selection
def SPEC_debris_combined_v11_50(json_file_name, json_input_dict):

	"""
	SPEC-debris incoporates multiprocessing v1.00

	Creator: Enok Cheon
	Date: 2022-03-13
	Language: Python3
	License: MIT

	perform SPEC-debris simulation for the following purpose:
		1) debris-flow propagation (with and without barrier)
		2) barrier performance against debris-flow collision

	Entrainment model options:
		'Hungr': E(s) = exp(Es*ds)
		'Er': E(s) = Es*ds

	Explanation of the json input file format for SPEC-debris are described in the user manual
	"""

	##################################################################################################################
	## read json file and extract input file data
	##################################################################################################################

	try: # exception for json input file error

		###########################################################
		## Name and folder information
		###########################################################
		# project name / export name
		exportName = json_input_dict["project name"] 

		# project folder path
		folder_path = json_input_dict["folder path"]
		if folder_path == None:  # current multiple json file folder path == each simulation folder path
			folder_path = os.path.dirname(json_file_name)+'/'
		else:
			folder_path = folder_path+'/'

		###########################################################
		## map data - DEM, road, wall/building
		###########################################################
		# DEM data
		flowpath_file_name = json_input_dict["flowpath file name"]
		source_file_name = json_input_dict["source file name"]
		material_file_name = json_input_dict["material file name"] 

		# road and goal XY-coordinates
		road_xy_list = [tuple(xy_pt) for xy_pt in json_input_dict["road xy"]]

		###########################################################
		## SPEC-debris analysis set-up and material properties
		###########################################################
		# time step parameters
		t_step = json_input_dict["time step interval"]
		t_max = json_input_dict["maximum simulation time"]

		# particle number
		if json_input_dict["particle number per cell"] is None:
			part_num_per_cell = 1
		else:
			try:
				part_num_per_cell = int(json_input_dict["particle number per cell"])
			except:
				part_num_per_cell = 1

		# initial setup of debris-flow - initial velocity
		initial_velocity = (json_input_dict["initial u_x"], json_input_dict["initial u_y"])

		# material
		material = {}
		for mat_key, mat_info in json_input_dict["material"].items():
			material[int(mat_key)] = mat_info

		# load - gravity
		g = json_input_dict["gravitational acceleration"]

		# interaction - coefficient of restitution COR
		COR = (json_input_dict["coefficient of restitution (COR)"]["particle with particle"], 
				json_input_dict["coefficient of restitution (COR)"]["particle with wall"])
		# COR = (1.0, json_input_dict["coefficient of restitution (COR)"]["particle with wall"]) 

		# perform SPH interpolation at time step = 0
		initial_SPH = json_input_dict["initial SPH"] 
  
		###########################################################
		## Algorithm options 
		###########################################################
		# interpolation method
		interp_method = json_input_dict["interpolation method"]

		# Entrainment model
		entrainment_model = json_input_dict["entrainment model"]

		# cluster boundary algorithm
		cluster_boundary = json_input_dict["cluster boundary method"]	

		###########################################################
		## Control parameters 
		###########################################################
		# no entrainment when climbing wall or free-falling
		Es_theta_var = json_input_dict["free fall angle variation"]

		# simulation coefficients and constants
		cell_size = tuple(json_input_dict["local cell sizes"]) 		

		# SPH smoothing length computation
		l_dp_min = json_input_dict["l_dp_min"]

		# cluster boundary control parameters for Concave-Hull
		concave_hull_parameter_dict = json_input_dict["concave hull algorithm parameter"]

		# mutliple cluster merge
		merge_overlap_ratio = json_input_dict["merge overlap ratio"]
  
		# multiprocessing
		max_cpu_num = json_input_dict["max cpu num"]

		###########################################################
		## optimal barrier location options - closed-type only
		###########################################################

		# selection of optimal barrier parameters 
		opt_total_loop = json_input_dict["optimal barrier location selection option"]["optimal total loop"]
		opt_avoid_dfl = json_input_dict["optimal barrier location selection option"]["optimal avoid dfl"]
		opt_iter_max = json_input_dict["optimal barrier location selection option"]["optimal iteration max"]
		opt_find_iter_limit = json_input_dict["optimal barrier location selection option"]["optimal find iteration limit"]
		opt_barrier_num_limit = [json_input_dict["optimal barrier location selection option"]["optimal barrier num min"],
									json_input_dict["optimal barrier location selection option"]["optimal barrier num max"]]

		# track cost per epoch
		output_optimal_step = json_input_dict["output optimal step"]

		# mixed both open and closed-type barrier
		opt_weight_combined = [
			json_input_dict["combined barrier optimization criteria"]["volume(V)"],
			json_input_dict["combined barrier optimization criteria"]["pressure(P)"],
			json_input_dict["combined barrier optimization criteria"]["distance_from_road(D)"],
			json_input_dict["combined barrier optimization criteria"]["closed_barrier_number(NC)"],
			json_input_dict["combined barrier optimization criteria"]["opened_barrier_number(NO)"]
		]

		# assumed barrier performance for optimal opened-type barriers
		open_performance = [
			json_input_dict["open barrier performance"]["speed_ratio(SR)"],
			json_input_dict["open barrier performance"]["trap_ratio(TR)"]
		]

		# vulnerability index
		min_uV = [0.0, 0.0]
		VI_crit = json_input_dict["critical vulnerability index (VI)"] 
		RC_bool = json_input_dict["reinforced concrete (RC) wall"]
		
		###########################################################
		## plotting options 
		###########################################################
		
		# plotting options
		plot_map_2D = json_input_dict["static plot option"]["plot map 2D"]
		plot_map_3D = json_input_dict["static plot option"]["plot map 3D"]
				
		# animation options
		plot_animation_2D = json_input_dict["animation option"]["plot animation 2D"]
		plot_animation_3D = json_input_dict["animation option"]["plot animation 3D"]
		plot_animation_2D_boundary = json_input_dict["animation option"]["plot animation 2D boundary"]

		# open plot after creating
		open_plot = json_input_dict["open plot"]

		# 3D plot option
		plot_3D_z_offset = json_input_dict["static plot option"]["plot 3D z offset"]

		# animation options
		animation = [
			json_input_dict["animation option"]["frame duration"], 
			json_input_dict["animation option"]["frame transition"], 
			json_input_dict["animation option"]["contour elevation interval"]
		]
		animation_3D_z_offset = json_input_dict["animation option"]["animation 3D z offset"]

		# plot maximum legend
		plot_2D_max_limits = [
			json_input_dict["max parameter legend"]["cluster max velocity(u)"],
			json_input_dict["max parameter legend"]["cluster max depth(h)"],
			json_input_dict["max parameter legend"]["cluster max volume(V)"],
			json_input_dict["max parameter legend"]["cluster max distance_from_road(D)"],
			json_input_dict["max parameter legend"]["cluster max pressure(P)"],
			json_input_dict["max parameter legend"]["particle max velocity(u)"],
			json_input_dict["max parameter legend"]["particle max depth(h)"],
			json_input_dict["max parameter legend"]["particle max volume(V)"],
			json_input_dict["max parameter legend"]["particle max pressure(P)"]
		]
		plot_3D_max_limits = deepcopy(plot_2D_max_limits)

		# marker, line, and plot size options
		marker_size = json_input_dict["marker size"]
		line_width = json_input_dict["line width"]
		layout_width = json_input_dict["layout width"]
		layout_height = json_input_dict["layout height"]

		###########################################################
		## output options 
		###########################################################
		# csv data export for SPEC-debris analysis
		csv_output = json_input_dict["csv output"]

		# data decimal places
		dp = json_input_dict["decimal points"]

	except (KeyError, NameError):
		print()
		print('json variable(s) name or value are missing or incorrect; please check input json file')
		print()
		return None

	##################################################################################################################
	## import data (las or grd or asc -> csv) for flowpath_file_name, source_file_name, and material_file_name
	##################################################################################################################

	try: # input file error

		##############################################################
		## flowpath_file_name
		##############################################################
		# find file type [csv, las, grd]
		flowpath_file_name_list = flowpath_file_name.split('.')
		flowpath_file_name_type = flowpath_file_name_list[-1]

		if flowpath_file_name_type == 'csv':
			pass
		
		elif flowpath_file_name_type in ['las', 'grd', 'asc']:

			# find the file name only
			flowpath_file_name_only_list = flowpath_file_name.split("."+flowpath_file_name_type)
			flowpath_file_name_only_list2 = [txt if n==0 or n==len(flowpath_file_name_only_list)-1 else flowpath_file_name_only_list2[n-1]+"."+flowpath_file_name_type+txt for n,txt in enumerate(flowpath_file_name_only_list)]
			flowpath_file_name_only = flowpath_file_name_only_list2[-2]

			# create xyz data and convert to csv data
			if flowpath_file_name_type == 'las':
				flowpath_database = las2xyz(folder_path+flowpath_file_name_only, outFileName=folder_path+flowpath_file_name_only, outFileFormat='csv', saveOutputFile=True)

			elif flowpath_file_name_type == 'grd':
				flowpath_database = grd2xyz(folder_path+flowpath_file_name_only, headDataOutput=False, outFileName=folder_path+flowpath_file_name_only, saveOutputFile=True)
		
			elif flowpath_file_name_type == 'asc':
				flowpath_database = asc2xyz(folder_path+flowpath_file_name_only, outFileName=folder_path+flowpath_file_name_only, saveOutputFile=True)

			flowpath_file_name = folder_path+flowpath_file_name_only+'.csv'

		else:
			raise RuntimeError

		##############################################################
		## source_file_name
		##############################################################
		# find file type [csv, las, grd]
		source_file_name_list = source_file_name.split('.')
		source_file_name_type = source_file_name_list[-1]

		if source_file_name_type == 'csv':
			pass
		
		elif source_file_name_type in ['las', 'grd', 'asc']:

			# find the file name only
			source_file_name_only_list = source_file_name.split("."+source_file_name_type)
			source_file_name_only_list2 = [txt if n==0 or n==len(source_file_name_only_list)-1 else source_file_name_only_list2[n-1]+"."+source_file_name_type+txt for n,txt in enumerate(source_file_name_only_list)]
			source_file_name_only = source_file_name_only_list2[-2]

			# create xyz data and convert to csv data
			if source_file_name_type == 'las':
				source_database = las2xyz(folder_path+source_file_name_only, outFileName=folder_path+source_file_name_only, outFileFormat='csv', saveOutputFile=True)

			elif source_file_name_type == 'grd':
				source_database = grd2xyz(folder_path+source_file_name_only, headDataOutput=False, outFileName=folder_path+source_file_name_only, saveOutputFile=True)

			elif source_file_name_type == 'asc':
				source_database = asc2xyz(folder_path+source_file_name_only, outFileName=folder_path+source_file_name_only, saveOutputFile=True)

			source_file_name = folder_path+source_file_name_only+'.csv'

		else:
			raise RuntimeError

		##############################################################
		## material_file_name
		##############################################################
		# import material file if number of number of material is higher than 1
		if material_file_name != None and len(material.keys()) > 1:
			# find file type [csv, las, grd]
			material_file_name_list = material_file_name.split('.')
			material_file_name_type = material_file_name_list[-1]

			if material_file_name_type == 'csv':
				pass
			
			elif material_file_name_type in ['las', 'grd', 'asc']:

				# find the file name only
				material_file_name_only_list = source_file_name.split("."+material_file_name_type)
				material_file_name_only_list2 = [txt if n==0 or n==len(material_file_name_only_list)-1 else material_file_name_only_list2[n-1]+"."+material_file_name_type+txt for n,txt in enumerate(material_file_name_only_list)]
				material_file_name_only = material_file_name_only_list2[-2]

				# create xyz data and convert to csv data
				if material_file_name_type == 'las':
					material_database = las2xyz(folder_path+material_file_name_only, outFileName=folder_path+material_file_name_only, outFileFormat='csv', saveOutputFile=True)

				elif material_file_name_type == 'grd':
					material_database = grd2xyz(folder_path+material_file_name_only, headDataOutput=False, outFileName=folder_path+material_file_name_only, saveOutputFile=True)

				elif material_file_name_type == 'asc':
					material_database = asc2xyz(folder_path+material_file_name_only, outFileName=folder_path+material_file_name_only, saveOutputFile=True)
			
				material_file_name = folder_path+material_file_name_only+'.csv'

			else:
				raise RuntimeError

		# import data issue
		elif material_file_name == None and len(material.keys()) > 1:
			raise KeyError

		elif len(material.keys()) == 1:
			pass

	except RuntimeError:
		print()
		print('please input csv or las (LiDAR) or grd (Surfer) for the flowpath/source/material file')
		print()
		return None
	except KeyError:
		print()
		print('please specify the material_file_name as number of material types > 1')
		print()
		return None

	##################################################################################################################
	## extract flowpath (elevation) and material data
	##################################################################################################################

	##############################################################
	## flowpath 
	##############################################################
	DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True) 

	##############################################################
	## eroding depth
	##############################################################
	ERODE = np.zeros(DEM_no_wall.shape)
 
	##############################################################
	## material
	##############################################################
	if material_file_name != None:
		MAT = xyz2mesh(folder_path+material_file_name, exportAll=False) 
	else:
		MAT = np.ones(DEM_no_wall.shape)

	##################################################################################################################
	## stable time step approach - without analyzing barrier performance
	##################################################################################################################

	# error message
	if road_xy_list == None:
		assert False, 'specify the road_xy_list to compute distance_from_road (D) factor'

	# parameters/variables related to wall
	wall_info = None
	wall_dict = None
	flowpath_file_name_with_wall = None
	DEM_with_wall = None
	wall_bound_region = None
	wall_perform_region = None
	wall_performance = None

	##############################################################
	## generate particle data at time step = 0
	##############################################################
	# generate particle and cluster data			
	cluster_list0, clusterID_flow_list0, max_cID_num, all_part_list0, part_num, part_radius = time0_setup_t_v12_0(folder_path+source_file_name, folder_path+flowpath_file_name, part_num_per_cell, cell_size[0], DEM_no_wall, MAT, material, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, road_xy_list, initial_velocity, g, wall_info, cluster_boundary, concave_hull_parameter_dict, l_dp_min, t_step, DP=dp, particle_only=False, initial_SPH=initial_SPH)

	# generate overall output summary data for time = 0
	overall_output_raw = np.array([[part_i.time, part_i.si, part_i.x, part_i.y, part_i.z, part_i.ui, part_i.hi, part_i.Vi] for part_i in all_part_list0[0]])
	overall_output_mean = np.mean(overall_output_raw[:,:-1], axis=0).tolist()
	overall_output_sum = np.sum(overall_output_raw[:,-1], axis=0).tolist()
	overall_output_summary0 = [overall_output_mean+[overall_output_sum]]

	print()
	print('completed setting initial simulation setting:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## SPEC-debris simulation for each weight factor sets
	##############################################################
	
	## SPEC-debris for the selected weight coefficients
	# cluster_list_final, overall_output_summary, all_part_list, t_track = SPEC_debris_c_t_nb_MP_v9_0(DEM_no_wall, ERODE, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, cluster_list0, clusterID_flow_list0, max_cID_num, all_part_list0, overall_output_summary0, road_xy_list, material, COR, Es_theta_var, cell_size, part_radius, l_dp_min, g, merge_overlap_ratio, cluster_boundary, concave_hull_parameter_dict, interp_method, entrainment_model, t_step, t_max, max_cpu_num)
	cluster_list_final, overall_output_summary, all_part_list, t_track = SPEC_debris_c_t_nb_MP_v9_0(DEM_no_wall, ERODE, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, cluster_list0, clusterID_flow_list0, max_cID_num, all_part_list0, overall_output_summary0, road_xy_list, material, COR, Es_theta_var, cell_size, part_radius, l_dp_min, g, merge_overlap_ratio, cluster_boundary, concave_hull_parameter_dict, interp_method, entrainment_model, t_step, t_max, max_cpu_num, csv_output, exportName, folder_path, DP=dp)

	print()
	print('completed SPEC-debris simulation:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))
 
	##############################################################
	## simulation export
	##############################################################
	## sort data and export into csv file
	# store particles all data
	# store particles all data
	all_part_data = []  # 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'
	# everything_part_data = []
	max_recorded_part_uhP = [0, 0, 0]
	volume_part_track = [0, 0] # initial, final
	latest_t_stored = 0
	for time_step, part_list_it in enumerate(all_part_list):

		all_part_data_temp = []
		# everything_part_data_t = []
		for part_i in part_list_it:
			# store particle data
			all_part_data_temp.append(list(part_i.return_all_param()))
			# everything_part_data_t.append(list(part_i.return_everything()))

			## SPEC analysis results
			# initial particles - total volume
			if time_step == 0:
				volume_part_track[0] += part_i.Vi
			# final particles - total volume
			elif time_step >= len(all_part_list)-1:
				volume_part_track[1] += part_i.Vi

			# maximum recorded particle velocity
			if max_recorded_part_uhP[0] < part_i.ui:
				max_recorded_part_uhP[0] = part_i.ui

			# maximum recorded particle depth
			if max_recorded_part_uhP[1] < part_i.hi:
				max_recorded_part_uhP[1] = part_i.hi

			# maximum recorded particle impact pressure
			if max_recorded_part_uhP[2] < part_i.Pi:
				max_recorded_part_uhP[2] = part_i.Pi

		all_part_data.append(all_part_data_temp)
		# everything_part_data.append(everything_part_data_t)

	# store each cluster data
	cluster_summary_dict = {}	# key = cluster_id, value = [max(S), max(u), max(h), max(P), V_t0, V_final]
	for cID in range(max_cID_num+1):
		cluster_summary_dict[str(cID)] = {
			"cluster volume at initial": 0, 
			"cluster volume at final": 0, 
			"total travel distance": 0, 
			"max recorded cluster velocity": 0, 
			"max recorded cluster depth": 0, 
			"max recorded cluster pressure": 0
		}

	all_cluster_data = []  # 'cID,t,s,x,y,z,u,h,V,D,P,area,CCH_alpha,merged'
	all_cluster_boundary_poly_dict = {}	# contains external boundary through shapely polygon
	for cluster_it in cluster_list_final:

		all_cluster_data_temp = []
		# all_cluster_boundary_poly_temp = []
		latest_t_stored = 0
		for time_step, cluster_i in enumerate(cluster_it):

			all_cluster_data_temp.append(list(cluster_i.return_all()))

			## shapely polygon - boundary
			# all_cluster_boundary_poly_temp.append(cluster_i.boundary_polygon)

			if int(cluster_i.clusterID) not in all_cluster_boundary_poly_dict.keys():
				all_cluster_boundary_poly_dict[int(cluster_i.clusterID)] = {
					time_step: cluster_i.boundary_polygon
				}

			else:
				temp_bound_dict = deepcopy(all_cluster_boundary_poly_dict[int(cluster_i.clusterID)])
				temp_bound_dict[time_step] = cluster_i.boundary_polygon
				all_cluster_boundary_poly_dict[int(cluster_i.clusterID)] = deepcopy(temp_bound_dict)
				del temp_bound_dict

			## SPEC analysis results
			# initial total volume
			if time_step == 0:
				cluster_summary_dict[str(cluster_i.clusterID)]["cluster volume at initial"] = cluster_i.Vc

			# final total volume and max travel distance
			elif time_step == len(cluster_it)-1:
				cluster_summary_dict[str(cluster_i.clusterID)]["cluster volume at final"] = cluster_i.Vc
				cluster_summary_dict[str(cluster_i.clusterID)]["total travel distance"] = cluster_i.sc

			# maximum recorded particle velocity
			if cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster velocity"] < cluster_i.uc:
				cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster velocity"] = cluster_i.uc

			# maximum recorded particle depth
			if cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster depth"] < cluster_i.hc:
				cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster depth"] = cluster_i.hc

			# maximum recorded particle impact pressure
			if cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster pressure"] < cluster_i.Pc:
				cluster_summary_dict[str(cluster_i.clusterID)]["max recorded cluster pressure"] = cluster_i.Pc

		all_cluster_data.append(deepcopy(all_cluster_data_temp))
		# all_cluster_boundary_poly_list.append(deepcopy(all_cluster_boundary_poly_temp))				
	## clusters - each cluster data - must be exported
	header_cluster = 'cID,t,s,x,y,z,u,h,V,D,P,A,Fr,CCHa,merged'
	cluster_summary_name_list = []
	for cID, loop_cluster_list in enumerate(all_cluster_data):
		cluster_summary_name = exportName+'_cluster'+str(cID)+'.csv'
		cluster_summary_name_list.append(cluster_summary_name)
		np.savetxt(folder_path+cluster_summary_name, loop_cluster_list, fmt='%.4f', delimiter=',', comments='', header=header_cluster)

	# store overall data
	all_output_summary_data = []	# 't,s,x,y,z,u,h,V'
	latest_t_stored = 0
	for overall_list_it in overall_output_summary:
		all_output_summary_data.append(deepcopy(overall_list_it))

	if csv_output:

		if exportName == None:
			exportName = 'SPEC-debris'
   
		## overall output - overall all particle summary data
		header_output = 't,s,x,y,z,u,h,V'
		np.savetxt(folder_path+exportName+'_overall_output.csv', all_output_summary_data, fmt='%.4f', delimiter=',', comments='', header=header_output)
	
	print()
	print('completed tabulating simulation results:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))
 
	##############################################################
	## simulation plotting
	##############################################################

	## 2D - plot static maps
	if plot_map_2D:
		plot_SPEC_debris_map_v6_0(folder_path, flowpath_file_name, all_part_data, all_cluster_data, all_output_summary_data, road_xy_list, exportName, max_limits=plot_2D_max_limits, open_html=open_plot, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height, max_cpu_num=max_cpu_num)

		print()
		print('completed 2D plotting simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## 2D - plot animation
	if plot_animation_2D:
		plot_SPEC_debris_animation_2D_plotly_v6_0(folder_path, flowpath_file_name, all_part_data, all_cluster_data, all_cluster_boundary_poly_dict, exportName, wall_dict, plot_animation_2D_boundary, animation_duration=animation[0], animation_transition=animation[1], contour_diff=animation[2], max_limits=plot_2D_max_limits, open_html=open_plot, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

		print()
		print('completed 2D animating simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## 3D - plot static maps
	if plot_map_3D:
		plot_SPEC_debris_surface_v4_0(folder_path, flowpath_file_name, all_part_data, all_cluster_data, exportName, max_limits=plot_3D_max_limits, open_html=open_plot, z_offset=plot_3D_z_offset, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

		print()
		print('completed 3D plotting simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))
		
	## 3D - plot animation	
	if plot_animation_3D:
		plot_SPEC_debris_animation_3D_plotly_v5_0(folder_path, flowpath_file_name, all_part_data, all_cluster_data, exportName, animation_duration=animation[0], animation_transition=animation[1], contour_diff=animation[2], max_limits=plot_3D_max_limits, open_html=open_plot, z_offset=animation_3D_z_offset, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

		print()
		print('completed 3D animating simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	print()
	print('completed plotting simulation results:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## optimal barrier location
	##############################################################
	# comptue fb_0 equivalent to f = (fb, ft) at computing energy loss (dE_loss)
	most_common_material_mode, most_common_material_count = mode(MAT)
	most_common_material_mode_list = most_common_material_mode.tolist()
	most_common_material_mode_list2 = most_common_material_mode_list[0]
	most_common_material_count_list = most_common_material_count.tolist()
	most_common_material_count_list2 = most_common_material_count_list[0]

	highest_count_idx = most_common_material_count_list2.index(max(most_common_material_count_list2))
	most_common_material_id = most_common_material_mode_list2[highest_count_idx]

	most_common_material_f = material[int(most_common_material_id)]["f"]
	fb_0 = comptue_fb_0_all_cluster_data(most_common_material_f, all_cluster_data)
	
	# find optimal open-type barrier locations
	optimal_dfl_combined_barrier_set, combined_barrier_cost, combined_barrier_dfl_data, terminus_dfl_data = iterate_find_optimal_combined_barrier_v3_0(folder_path, flowpath_file_name, cluster_list_final, road_xy_list, material, MAT, DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, fb_0, alpha=1.5, total_loop=opt_total_loop, open_performance=open_performance, opt_weight=opt_weight_combined, avoid_dfl=opt_avoid_dfl, iteration_limit=opt_iter_max, find_iter_limit=opt_find_iter_limit, cell_size=cell_size[0], g=g, entrainment_model=entrainment_model, Es_theta_var=Es_theta_var, interp_method=interp_method, min_uV=min_uV, VI_crit=VI_crit, RC_bool=RC_bool, barrier_num_limit=opt_barrier_num_limit, output_step=output_optimal_step, plot_network=plot_map_2D, exportName=exportName, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height, max_cpu_num=max_cpu_num, dp=dp)

	print()
	print('completed searching for optimal combined open- and closed-type barriers:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))
 
 
	# combined_barrier_dfl_data = {} # key = dfl, value = [barrier_performance, [cID_list], [V, P, D] [normalized V, P, D]
	combined_barrier_dfl_data_mod = {}
	for DFLi, data_i in combined_barrier_dfl_data.items():
		temp_list = deepcopy(data_i)
  
		# create class for writing optimal dfl
		temp_class_result = {
			"barrier performance": temp_list[0],
			"cluser ID": temp_list[1],
			"optimization criteria": {
				"volume(V)": temp_list[2][0], 
				"pressure(P)": temp_list[2][1],
				"distance_from_road(D)": temp_list[2][2]
			},
			"normalized optimization criteria": {
				"normalized volume(V)": temp_list[3][0], 
				"normalized pressure(P)": temp_list[3][1], 
				"normalized distance_from_road(D)": temp_list[3][2]
			}
		}

		combined_barrier_dfl_data_mod[DFLi] = deepcopy(temp_class_result)
		del temp_class_result, temp_list
  
	# dfl_terminus_VI = {}  # {terminus_dfl : [velocity, VI]}
	dfl_terminus_VI_mod = {} 
	for tDFLi, data_i in terminus_dfl_data.items():
		temp_list = deepcopy(data_i)
  
		# create class for writing optimal dfl
		temp_class_result = {
			"velocity at terminus": temp_list[0],
			"vulnerability index (VI) from velocity": temp_list[1]
		}

		dfl_terminus_VI_mod[tDFLi] = deepcopy(temp_class_result)
		del temp_class_result, temp_list
 
	now = datetime.now()
	current_output_date_time = now.strftime("%Y/%m/%d %H:%M:%S")

	temp_output = {
		# heading info
		'time-date': current_output_date_time,
		'project name': json_input_dict["project name"],  # project name / export name
		'folder path': folder_path,	# project folder path

		# DEM data
		"flowpath file name": json_input_dict["flowpath file name"], 
		'flowpath file name with wall': flowpath_file_name_with_wall, 
		"source file name": json_input_dict["source file name"],
		"material file name": json_input_dict["material file name"],

		# road and goal XY-coordinates
		'road xy': str(json_input_dict["road xy"]),

		# initial setup of debris-flow - initial velocity and particle number
		'initial_velocity': {
			"initial u_x": json_input_dict["initial u_x"], 
			"initial u_y": json_input_dict["initial u_y"]
		},
		"particle number": part_num,

		'material': json_input_dict["material"], # material
		'gravitational acceleration': g,		# load - gravity
		"coefficient of restitution (COR)":{
			"particle with particle": COR[0],
			"particle with wall": COR[1]
		},

		# no entrainment when climbing wall or free-falling
		'free fall angle variation': Es_theta_var,

		# pathway weight factor
		'local cell sizes': str(json_input_dict["local cell sizes"]),

		# interpolation method
		'interpolation method': json_input_dict["interpolation method"],

		# Entrainment model
		'entrainment model': json_input_dict["entrainment model"],

		# mutliple cluster merge
		'merge overlap ratio': json_input_dict["merge overlap ratio"],
		
		# cluster boundary
		'cluster boundary method': cluster_boundary,
		'concave hull algorithm parameter': None if cluster_boundary=="ConvexHull" else concave_hull_parameter_dict,

		## simulation for time-step and barrier performance
		# time step parameters
		"time step interval": t_step,
		"maximum simulation time": round(t_track, dp), 

		# SPEC analysis results 
		"particle summary": {
			"total initial volume": volume_part_track[0],
			"total final volume": volume_part_track[1],
			"max recorded particle velocity": max_recorded_part_uhP[0],
			"max recorded particle depth": max_recorded_part_uhP[1],
			"max recorded particle impact pressure": max_recorded_part_uhP[2]
		},
		
		# SPEC cluster analysis results 
		"cluster summary": cluster_summary_dict,
  
		# optimization search options
		"optimal barrier location selection option": json_input_dict["optimal barrier location selection option"],
		"combined barrier optimization criteria": json_input_dict["combined barrier optimization criteria"],
		"open barrier performance": json_input_dict["open barrier performance"],
		"critical vulnerability index (VI)":json_input_dict["critical vulnerability index (VI)"],
		"reinforced concrete (RC) wall": json_input_dict["reinforced concrete (RC) wall"], 
	
		# optimal combined (closed and open) barrier 
		"optimal barrier location selection summary": {
			"optimal closed-type barrier locations": optimal_dfl_combined_barrier_set[0],
			"optimal open-type barrier locations": optimal_dfl_combined_barrier_set[1],
			"optimal cost": combined_barrier_cost,
			"optimal DFL data": combined_barrier_dfl_data_mod,
			"terminus DFL data": dfl_terminus_VI_mod
		}
	}

	# create an output json file
	with open(folder_path+exportName+'.json', 'w') as fp:
		json.dump(temp_output, fp, indent=4)

	print()
	print('completed creating summary JSON file:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## export json files with all inputs so that only the optimal barrier location can be run

	# convert array to list
	MAT_list = MAT.tolist()
	DEM_no_wall_list = DEM_no_wall.tolist()
	gridUniqueX_list = gridUniqueX.tolist()
	gridUniqueY_list = gridUniqueY.tolist()

	opt_combined_output = {
		'combined': [
			{
				"project name": exportName,
				'folder path': folder_path,
				'flowpath file name': json_input_dict["flowpath file name"], 
				'road xy': json_input_dict["road xy"],

				'material': json_input_dict["material"],
				'material id array': MAT_list,
				'flowpath z array': DEM_no_wall_list,
				'grid x array': gridUniqueX_list,
				'grid y array': gridUniqueY_list,
				'grid delta_x': deltaX,
				'grid delta_y': deltaY,
				'overall_fb_0': fb_0,

				"optimal barrier location selection option":{
					"optimal total loop": opt_total_loop, 
					"optimal avoid dfl": opt_avoid_dfl, 
					"optimal iteration max": opt_iter_max, 
					"optimal find iteration limit": opt_find_iter_limit, 
					"optimal barrier num min": opt_barrier_num_limit[0], 
					"optimal barrier num max": opt_barrier_num_limit[1]
				}, 
				"combined barrier optimization criteria": {
					"volume(V)": opt_weight_combined[0], 
					"pressure(P)": opt_weight_combined[1],
					"distance_from_road(D)": opt_weight_combined[2],
					"closed_barrier_number(NC)": opt_weight_combined[3],
					"opened_barrier_number(NO)": opt_weight_combined[4]
				}, 
				"open barrier performance": {
					"speed_ratio(SR)": open_performance[0],
					"trap_ratio(TR)": open_performance[1]
				}, 
				"critical vulnerability index (VI)": VI_crit, 
				"reinforced concrete (RC) wall": RC_bool, 
				'output optimal step': output_optimal_step,

				'local cell sizes': cell_size,
				'gravitational acceleration': g,
				'entrainment model': entrainment_model,
				'free fall angle variation': Es_theta_var,
				'interpolation method': interp_method,

				"static plot option": {
					"plot map 2D": plot_map_2D,
					"plot map 3D": plot_map_3D,
					"plot 3D z offset": plot_3D_z_offset
				},
				'marker size': marker_size,
				'line width': line_width,
				'layout width': layout_width,
				'layout height': layout_height,

				'max cpu num': max_cpu_num,
				'decimal points': dp,

				'SPEC_debris analysis cluster data': cluster_summary_name_list
			}
		]
	}

	# create an output json file
	with open(folder_path+exportName+' - opt_combined_input.json', 'w') as fp:
		json.dump(opt_combined_output, fp, indent=4)

	print()
	print('completed generating input JSON file for optimal combined open- and closed-type barrier location search:*********************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))
 
	# if everything goes well - return code to signify all completed well
	return temp_output     

# SPEC-debris simulation
def SPEC_debris_v11_50(json_file_name, json_input_dict, wall_keyword=False):

	"""
	SPEC-debris incoporates multiprocessing v1.00

	Creator: Enok Cheon
	Date: 2022-03-13
	Language: Python3
	License: MIT

	perform SPEC-debris simulation for the following purpose:
		1) debris-flow propagation (with and without barrier)
		2) barrier performance against debris-flow collision

	Entrainment model options:
		'Hungr': E(s) = exp(Es*ds)
		'Er': E(s) = Es*ds

	Explanation of the json input file format for SPEC-debris are described in the user manual
	"""

	try: # exception for json input file error

		###########################################################
		## Name and folder information
		###########################################################
		# project name / export name
		exportName = json_input_dict["project name"] 

		# project folder path
		folder_path = json_input_dict["folder path"]
		if folder_path == None:  # current multiple json file folder path == each simulation folder path
			folder_path = os.path.dirname(json_file_name)+'/'
		else:
			folder_path = folder_path+'/'

		###########################################################
		## map data - DEM and road
		###########################################################
		# DEM data
		flowpath_file_name = json_input_dict["flowpath file name"]
		source_file_name = json_input_dict["source file name"]
		material_file_name = json_input_dict["material file name"] 

		# road and goal XY-coordinates
		road_xy_list = [tuple(xy_pt) for xy_pt in json_input_dict["road xy"]]

		###########################################################
		## map data - wall/building
		###########################################################
		if json_input_dict["wall info"] == None:
			wall_info = None
		else:
			wall_info = []
			for wall_group_num_str, wall_i_info in json_input_dict["wall info"].items():

				# for any box-shaped barriers [closed or parallel (P) slit or V-shaped (V) slit or baffles]
				# wall_info -> [wall_group_id, type ('P' or 'V'), slit_ratio, wall_segment_number, wall_segment_oriP (-90 ~ 90), wall_oriP (-90 ~ 90), 
				# 	thickness, length, Z_opt (1~4), height/elevation, central_X_coord, central_Y_coord]
				if wall_i_info["wall type"] in ["P", "V"]:
					wall_info.append([
						int(wall_group_num_str),
						wall_i_info["wall type"], 
						wall_i_info["slit ratio"], 
						wall_i_info["number of wall segments"], 
						wall_i_info["orientation of wall segments (Polar)"], 
						wall_i_info["orientation of wall overall (Polar)"], 
						wall_i_info["wall thickness"], 
						wall_i_info["wall length"], 
						wall_i_info["wall elevation option"], 
						wall_i_info["elevation or height"], 
						wall_i_info["wall centroid X"], 
						wall_i_info["wall centroid Y"]
					])

				# circle-shaped barriers [circular baffles]
				# wall_info -> [wall_group_id, type ('C'), cylinder_number, wall_oriP (-90 ~ 90), radius, length, Z_opt (1~4), height/elevation, central_X_coord, central_Y_coord]
				elif wall_i_info["wall type"] == "C":
					wall_info.append([
						int(wall_group_num_str),
						wall_i_info["wall type"], 
						wall_i_info["number of cylinder segments"], 
						wall_i_info["orientation of wall overall (Polar)"], 
						wall_i_info["cylinder radius"], 
						wall_i_info["wall length"], 
						wall_i_info["wall elevation option"], 
						wall_i_info["elevation or height"], 
						wall_i_info["wall centroid X"], 
						wall_i_info["wall centroid Y"]
					])

				# for user defined shape barriers
				# wall_info -> [wall_group_id, Type('BD'), Z_opt, height/elevation, XY_list ] 
				elif wall_i_info["wall type"] == "BD":
					wall_info.append([
						int(wall_group_num_str),
						wall_i_info["wall type"], 
						wall_i_info["wall elevation option"], 
						wall_i_info["elevation or height"], 
						wall_i_info["wall XY points"]
					])

		###########################################################
		## SPEC-debris analysis set-up and material properties
		###########################################################
		# time step parameters
		t_step = json_input_dict["time step interval"]
		t_max = json_input_dict["maximum simulation time"]

		# particle number
		if json_input_dict["particle number per cell"] is None:
			part_num_per_cell = 1
		else:
			try:
				part_num_per_cell = int(json_input_dict["particle number per cell"])
			except:
				part_num_per_cell = 1

		# initial setup of debris-flow - initial velocity
		initial_velocity = (json_input_dict["initial u_x"], json_input_dict["initial u_y"])

		# material
		material = {}
		for mat_key, mat_info in json_input_dict["material"].items():
			material[int(mat_key)] = mat_info

		# load - gravity
		g = json_input_dict["gravitational acceleration"]

		# interaction - coefficient of restitution COR
		COR = (json_input_dict["coefficient of restitution (COR)"]["particle with particle"], 
				json_input_dict["coefficient of restitution (COR)"]["particle with wall"])
		# COR = (1.0, json_input_dict["coefficient of restitution (COR)"]["particle with wall"]) 

		# perform SPH interpolation at time step = 0
		initial_SPH = json_input_dict["initial SPH"] 

		###########################################################
		## Algorithm options 
		###########################################################
		# interpolation method
		interp_method = json_input_dict["interpolation method"]

		# Entrainment model
		entrainment_model = json_input_dict["entrainment model"]

		# cluster boundary algorithm
		cluster_boundary = json_input_dict["cluster boundary method"]	

		###########################################################
		## Control parameters 
		###########################################################
		# no entrainment when climbing wall or free-falling
		Es_theta_var = json_input_dict["free fall angle variation"]

		# simulation coefficients and constants
		cell_size = tuple(json_input_dict["local cell sizes"]) 		

		# SPH smoothing length computation
		l_dp_min = json_input_dict["l_dp_min"]
  
		# multiprocessing
		max_cpu_num = json_input_dict["max cpu num"]
		
		###########################################################
		## plotting options 
		###########################################################
		
		# plotting options
		plot_map_2D = json_input_dict["static plot option"]["plot map 2D"]
		plot_map_3D = json_input_dict["static plot option"]["plot map 3D"]
				
		# animation options
		plot_animation_2D = json_input_dict["animation option"]["plot animation 2D"]
		plot_animation_3D = json_input_dict["animation option"]["plot animation 3D"]
		plot_animation_2D_boundary = json_input_dict["animation option"]["plot animation 2D boundary"]

		# open plot after creating
		open_plot = json_input_dict["open plot"]

		# 3D plot option
		plot_3D_z_offset = json_input_dict["static plot option"]["plot 3D z offset"]

		# animation options
		animation = [
			json_input_dict["animation option"]["frame duration"], 
			json_input_dict["animation option"]["frame transition"], 
			json_input_dict["animation option"]["contour elevation interval"]
		]
		animation_3D_z_offset = json_input_dict["animation option"]["animation 3D z offset"]

		# plot maximum legend
		plot_2D_max_limits = [
			json_input_dict["max parameter legend"]["cluster max velocity(u)"],
			json_input_dict["max parameter legend"]["cluster max depth(h)"],
			json_input_dict["max parameter legend"]["cluster max volume(V)"],
			json_input_dict["max parameter legend"]["cluster max distance_from_road(D)"],
			json_input_dict["max parameter legend"]["cluster max pressure(P)"],
			json_input_dict["max parameter legend"]["particle max velocity(u)"],
			json_input_dict["max parameter legend"]["particle max depth(h)"],
			json_input_dict["max parameter legend"]["particle max volume(V)"],
			json_input_dict["max parameter legend"]["particle max pressure(P)"]
		]
		plot_3D_max_limits = deepcopy(plot_2D_max_limits)

		# marker, line, and plot size options
		marker_size = json_input_dict["marker size"]
		line_width = json_input_dict["line width"]
		layout_width = json_input_dict["layout width"]
		layout_height = json_input_dict["layout height"]

		###########################################################
		## output options 
		###########################################################
		# csv data export for SPEC-debris analysis
		csv_output = json_input_dict["csv output"]

		# data decimal places
		dp = json_input_dict["decimal points"]

	except (KeyError, NameError):
		print()
		print('json variable(s) name or value are missing or incorrect; please check input json file')
		print()
		return None

	##################################################################################################################
	## import data (las or grd or asc -> csv) for flowpath_file_name, source_file_name, and material_file_name
	##################################################################################################################

	try: # input file error

		##############################################################
		## flowpath_file_name
		##############################################################
		# find file type [csv, las, grd]
		flowpath_file_name_list = flowpath_file_name.split('.')
		flowpath_file_name_type = flowpath_file_name_list[-1]

		if flowpath_file_name_type == 'csv':
			pass
		
		elif flowpath_file_name_type in ['las', 'grd', 'asc']:

			# find the file name only
			flowpath_file_name_only_list = flowpath_file_name.split("."+flowpath_file_name_type)
			flowpath_file_name_only_list2 = [txt if n==0 or n==len(flowpath_file_name_only_list)-1 else flowpath_file_name_only_list2[n-1]+"."+flowpath_file_name_type+txt for n,txt in enumerate(flowpath_file_name_only_list)]
			flowpath_file_name_only = flowpath_file_name_only_list2[-2]

			# create xyz data and convert to csv data
			if flowpath_file_name_type == 'las':
				flowpath_database = las2xyz(folder_path+flowpath_file_name_only, outFileName=folder_path+flowpath_file_name_only, outFileFormat='csv', saveOutputFile=True)

			elif flowpath_file_name_type == 'grd':
				flowpath_database = grd2xyz(folder_path+flowpath_file_name_only, headDataOutput=False, outFileName=folder_path+flowpath_file_name_only, saveOutputFile=True)
		
			elif flowpath_file_name_type == 'asc':
				flowpath_database = asc2xyz(folder_path+flowpath_file_name_only, outFileName=folder_path+flowpath_file_name_only, saveOutputFile=True)

			flowpath_file_name = folder_path+flowpath_file_name_only+'.csv'

		else:
			raise RuntimeError

		##############################################################
		## source_file_name
		##############################################################
		# find file type [csv, las, grd]
		source_file_name_list = source_file_name.split('.')
		source_file_name_type = source_file_name_list[-1]

		if source_file_name_type == 'csv':
			pass
		
		elif source_file_name_type in ['las', 'grd', 'asc']:

			# find the file name only
			source_file_name_only_list = source_file_name.split("."+source_file_name_type)
			source_file_name_only_list2 = [txt if n==0 or n==len(source_file_name_only_list)-1 else source_file_name_only_list2[n-1]+"."+source_file_name_type+txt for n,txt in enumerate(source_file_name_only_list)]
			source_file_name_only = source_file_name_only_list2[-2]

			# create xyz data and convert to csv data
			if source_file_name_type == 'las':
				source_database = las2xyz(folder_path+source_file_name_only, outFileName=folder_path+source_file_name_only, outFileFormat='csv', saveOutputFile=True)

			elif source_file_name_type == 'grd':
				source_database = grd2xyz(folder_path+source_file_name_only, headDataOutput=False, outFileName=folder_path+source_file_name_only, saveOutputFile=True)

			elif source_file_name_type == 'asc':
				source_database = asc2xyz(folder_path+source_file_name_only, outFileName=folder_path+source_file_name_only, saveOutputFile=True)

			source_file_name = folder_path+source_file_name_only+'.csv'

		else:
			raise RuntimeError

		##############################################################
		## material_file_name
		##############################################################
		# import material file if number of number of material is higher than 1
		if material_file_name != None and len(material.keys()) > 1:
			# find file type [csv, las, grd]
			material_file_name_list = material_file_name.split('.')
			material_file_name_type = material_file_name_list[-1]

			if material_file_name_type == 'csv':
				pass
			
			elif material_file_name_type in ['las', 'grd', 'asc']:

				# find the file name only
				material_file_name_only_list = source_file_name.split("."+material_file_name_type)
				material_file_name_only_list2 = [txt if n==0 or n==len(material_file_name_only_list)-1 else material_file_name_only_list2[n-1]+"."+material_file_name_type+txt for n,txt in enumerate(material_file_name_only_list)]
				material_file_name_only = material_file_name_only_list2[-2]

				# create xyz data and convert to csv data
				if material_file_name_type == 'las':
					material_database = las2xyz(folder_path+material_file_name_only, outFileName=folder_path+material_file_name_only, outFileFormat='csv', saveOutputFile=True)

				elif material_file_name_type == 'grd':
					material_database = grd2xyz(folder_path+material_file_name_only, headDataOutput=False, outFileName=folder_path+material_file_name_only, saveOutputFile=True)

				elif material_file_name_type == 'asc':
					material_database = asc2xyz(folder_path+material_file_name_only, outFileName=folder_path+material_file_name_only, saveOutputFile=True)
			
				material_file_name = folder_path+material_file_name_only+'.csv'

			else:
				raise RuntimeError

		# import data issue
		elif material_file_name == None and len(material.keys()) > 1:
			raise KeyError

		elif len(material.keys()) == 1:
			pass

	except RuntimeError:
		print()
		print('please input csv or las (LiDAR) or grd (Surfer) for the flowpath/source/material file')
		print()
		return None
	except KeyError:
		print()
		print('please specify the material_file_name as number of material types > 1')
		print()
		return None

	##################################################################################################################
	## extract flowpath (elevation) and material data
	##################################################################################################################
	
	##############################################################
	## flowpath 
	##############################################################
	DEM_no_wall, gridUniqueX, gridUniqueY, deltaX, deltaY = xyz2mesh(folder_path+flowpath_file_name, exportAll=True) 
 
	##############################################################
	## eroding depth
	##############################################################
	ERODE = np.zeros(DEM_no_wall.shape)

	##############################################################
	## material
	##############################################################
	if material_file_name != None:
		MAT = xyz2mesh(folder_path+material_file_name, exportAll=False) 
	else:
		MAT = np.ones(DEM_no_wall.shape)

	##################################################################################################################
	## wall
	##################################################################################################################
	if wall_info == None:
		# parameters/variables related to wall
		wall_info = None
		wall_dict = None
		flowpath_file_name_with_wall = None
		plot_export_flowpath_file_name = flowpath_file_name
		DEM_with_wall = None
		wall_bound_region = None
		wall_perform_region = None

	elif wall_info != None: 

		## generate wall data and DEM with wall
		# add wall data into topography and as dictionary
		wall_dict = generate_wall_dict_v1_00(wall_info)
		
		# modify DEM to contain buildings 
		# similar to DTM but can be later recognize buildings or barriers for collision physics
		num = 0
		for type_wall_id, wall_data_list in wall_dict.items():

			## create flowpath that has the wall incoporated as the topography
			for wall_segement_data, wall_poly in zip(wall_data_list[1], wall_data_list[2]):
				if num == 0: 
					modify_DEM_v1_0(folder_path+flowpath_file_name, wall_segement_data, wall_poly, outFileName=folder_path+exportName+'_flowpath_addWall', saveFileFormat='csv')
					num += 1
				else:
					modify_DEM_v1_0(folder_path+exportName+'_flowpath_addWall.csv', wall_segement_data, wall_poly, outFileName=folder_path+exportName+'_flowpath_addWall', saveFileFormat='csv')

		flowpath_file_name_with_wall = exportName+'_flowpath_addWall.csv'
		plot_export_flowpath_file_name = flowpath_file_name_with_wall

		DEM_with_wall = xyz2mesh(folder_path+flowpath_file_name_with_wall, exportAll=False)   

		wall_bound_region = compute_wall_surrounding_region_v2_0(wall_dict, DEM_with_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, min(cell_size), interp_method, SPH=True)
		wall_perform_region = compute_wall_surrounding_region_v2_0(wall_dict, DEM_with_wall, gridUniqueX, gridUniqueY, deltaX, deltaY, min(cell_size), interp_method, SPH=False)

	##############################################################
	## SPEC-debris
	##############################################################
	## generate particle data at time step = 0
	all_part_list0, part_num, part_radius = time0_setup_t_v12_0(folder_path+source_file_name, folder_path+flowpath_file_name, part_num_per_cell, cell_size[0], DEM_no_wall, MAT, material, gridUniqueX, gridUniqueY, deltaX, deltaY, interp_method, road_xy_list, initial_velocity, g, wall_info, cluster_boundary, None, l_dp_min, t_step, DP=dp, particle_only=True, initial_SPH=initial_SPH)

	print()
	print('completed setting initial simulation setting:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## SPEC-debris simulation
	# use adaptive weight factor
	all_part_list, t_track = SPEC_debris_c_t_MP_v9_0(wall_dict, wall_bound_region, DEM_no_wall, DEM_with_wall, ERODE, MAT, gridUniqueX, gridUniqueY, deltaX, deltaY, all_part_list0, material, COR, Es_theta_var, cell_size, part_radius, l_dp_min, g, interp_method, entrainment_model, t_step, t_max, max_cpu_num, csv_output, exportName, folder_path, DP=dp)
 
	print()
	print('completed SPEC-debris simulation:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## simulation export
	##############################################################

	## sort data and export into csv file
	# store particles all data
	all_part_data = []  # 'cID,t,s,x,y,z,elevation,u,ux,uy,h,V,P'
	# everything_part_data = []
	max_recorded_part_uhP = [0, 0, 0]
	volume_part_track = [0, 0] # initial, final
	# latest_t_stored = 0
	for time_step, part_list_it in enumerate(all_part_list):

		all_part_data_temp = []
		# everything_part_data_t = []
		for part_i in part_list_it:
			# store particle data
			all_part_data_temp.append(list(part_i.return_all_param()))
			# everything_part_data_t.append(list(part_i.return_everything()))

			## SPEC analysis results
			# initial particles - total volume
			if time_step == 0:
				volume_part_track[0] += part_i.Vi
			# final particles - total volume
			elif time_step >= len(all_part_list)-1:
				volume_part_track[1] += part_i.Vi

			# maximum recorded particle velocity
			if max_recorded_part_uhP[0] < part_i.ui:
				max_recorded_part_uhP[0] = part_i.ui

			# maximum recorded particle depth
			if max_recorded_part_uhP[1] < part_i.hi:
				max_recorded_part_uhP[1] = part_i.hi

			# maximum recorded particle impact pressure
			if max_recorded_part_uhP[2] < part_i.Pi:
				max_recorded_part_uhP[2] = part_i.Pi

		all_part_data.append(all_part_data_temp)
		# everything_part_data.append(everything_part_data_t)

	print()
	print('completed tabulating simulation results:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))
	
	##############################################################
	## simulation plotting
	##############################################################

	## 2D - plot static maps
	if plot_map_2D:
		plot_SPEC_debris_map_v6_0(folder_path, plot_export_flowpath_file_name, all_part_data, None, None, road_xy_list, exportName, max_limits=plot_2D_max_limits, open_html=open_plot, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height, max_cpu_num=max_cpu_num)

		print()
		print('completed 2D plotting simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))
	
	## 2D - plot animation
	if plot_animation_2D:
		plot_SPEC_debris_animation_2D_plotly_v6_0(folder_path, plot_export_flowpath_file_name, all_part_data, None, None, exportName, wall_dict, plot_animation_2D_boundary, animation_duration=animation[0], animation_transition=animation[1], contour_diff=animation[2], max_limits=plot_2D_max_limits, open_html=open_plot, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

		print()
		print('completed 2D animating simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## 3D - plot static maps
	if plot_map_3D:
		plot_SPEC_debris_surface_v4_0(folder_path, plot_export_flowpath_file_name, all_part_data, None, exportName, max_limits=plot_3D_max_limits, open_html=open_plot, z_offset=plot_3D_z_offset, marker_size=marker_size, line_width=line_width, layout_width=layout_width, layout_height=layout_height)

		print()
		print('completed 3D plotting simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	## 3D - plot animation	
	if plot_animation_3D:
		plot_SPEC_debris_animation_3D_plotly_v5_0(folder_path, plot_export_flowpath_file_name, all_part_data, None, exportName, animation_duration=animation[0], animation_transition=animation[1], contour_diff=animation[2], max_limits=plot_3D_max_limits, open_html=open_plot, z_offset=animation_3D_z_offset, marker_size=marker_size, line_width=line_width, layout_width=layout_width,layout_height=layout_height)

		print()
		print('completed 3D animating simulation results:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	print()
	print('completed plotting simulation results:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## wall performance evaluation
	##############################################################
	if wall_keyword == False:
		wall_performance = None

	elif wall_keyword == True:	

		# wall performance
		wall_performance = compute_performance_v2_0(wall_perform_region, all_part_list, max_cpu_num)

		# performance = [AR, TR] = [(av_U_after/av_U_before), (V_after/V_before)]
		# wall_performance[wall_id] = [(part_av_u_after/max_part_av_u_before), (part_sum_V_after/max_part_sum_V_before)]

		print()
		print('completed analyzing performance of opened-type barriers:****************************************************')
		now = datetime.now()
		print(now.strftime("%Y/%m/%d %H:%M:%S"))

	##############################################################
	## simulation outputs
	##############################################################
	
	now = datetime.now()
	current_output_date_time = now.strftime("%Y/%m/%d %H:%M:%S")

	temp_output = {
		# heading info
		'time-date': current_output_date_time,
		'project name': json_input_dict["project name"],  # project name / export name
		'folder path': folder_path,	# project folder path

		# DEM data
		"flowpath file name": json_input_dict["flowpath file name"], 
		'flowpath file name with wall': flowpath_file_name_with_wall, 
		"source file name": json_input_dict["source file name"],
		"material file name": json_input_dict["material file name"],

		# initial setup of debris-flow - initial velocity and particle number
		'initial_velocity': {
			"initial u_x": json_input_dict["initial u_x"], 
			"initial u_y": json_input_dict["initial u_y"]
		},
		"particle number": part_num,

		'material': json_input_dict["material"],	# material
		'gravitational acceleration': g,	# load - gravity
		"coefficient of restitution (COR)":{
			"particle with particle": COR[0],
			"particle with wall": COR[1]
		},

		# no entrainment when climbing wall or free-falling
		'free fall angle variation': Es_theta_var,

		# pathway weight factor
		'local cell sizes': json_input_dict["local cell sizes"],

		# interpolation method
		'interpolation method': json_input_dict["interpolation method"],

		# Entrainment model
		'entrainment model': json_input_dict["entrainment model"],

		## simulation for time-step and barrier performance
		# time step parameters
		"time step interval": t_step,
		"maximum simulation time": round(t_track, dp), 

		# SPEC particle analysis results 
		"particle summary": {
			"total initial volume": volume_part_track[0],
			"total final volume": volume_part_track[1],
			"max recorded particle velocity": max_recorded_part_uhP[0],
			"max recorded particle depth": max_recorded_part_uhP[1],
			"max recorded particle impact pressure": max_recorded_part_uhP[2]
		}
	}

	if wall_info != None:

		wall_information_dict = {}
		wall_segment_information_dict = {}
		
		# add wall overall and segment information
		for wall_id, wall_in in wall_dict.items():
			
			## wall overall information
			t_wall_i_dict = {}
			for wall_overall_info in wall_in[0]:

				# for any box-shaped barriers [closed or parallel (P) slit or V-shaped (V) slit or baffles]
				if wall_overall_info[1] in ["P", "V"]:
					t_wall_i_dict[int(wall_overall_info[0])] = {
						"wall type": wall_overall_info[1],
						"slit ratio": wall_overall_info[2],
						"number of wall segments": wall_overall_info[3],
						"orientation of wall segments (Polar)": wall_overall_info[4],
						"orientation of wall overall (Polar)": wall_overall_info[5],
						"wall thickness": wall_overall_info[6],
						"wall length": wall_overall_info[7],
						"wall elevation option": wall_overall_info[8], 
						"elevation or height": wall_overall_info[9],
						"wall centroid X": wall_overall_info[10],
						"wall centroid Y": wall_overall_info[11]
					}

				# circle-shaped barriers [circular baffles]
				elif wall_overall_info[1] == "C":
					t_wall_i_dict[int(wall_overall_info[0])] = {
						"wall type": wall_overall_info[1],
						"number of cylinder segments": wall_overall_info[2],
						"orientation of wall overall (Polar)": wall_overall_info[3],
						"cylinder radius": wall_overall_info[4],
						"wall length": wall_overall_info[5],
						"wall elevation option": wall_overall_info[6],
						"elevation or height": wall_overall_info[7],
						"wall centroid X": wall_overall_info[8],
						"wall centroid Y": wall_overall_info[9]
					}

				# for user defined shape barriers
				elif wall_overall_info[1] == "BD":
					t_wall_i_dict[int(wall_overall_info[0])] = {
						"wall type": wall_overall_info[1],
						"wall elevation option": wall_overall_info[2],
						"elevation or height": wall_overall_info[3],
						"wall XY points": wall_overall_info[4]
					}

			wall_information_dict[wall_id] = deepcopy(t_wall_i_dict)
			del t_wall_i_dict

			## input each segment data
			t_seg_dict = {}
			for wall_seg_in in wall_in[1]:

				if wall_seg_in[0] in ["P", "V"]:
					t_seg_dict[wall_seg_in[1]] = {
						'wall type': wall_seg_in[0],
						'wall seg id': wall_seg_in[1],
						'wall elevation option': wall_seg_in[2],
						'elevation or height': wall_seg_in[3],
						'corner1 xy points': wall_seg_in[4][0],
						'corner2 xy points': wall_seg_in[4][1],
						'corner3 xy points': wall_seg_in[4][2],
						'corner4 xy points': wall_seg_in[4][3]
					}

				elif wall_seg_in[0] == "C":
					t_seg_dict[wall_seg_in[1]] = {
						'wall type': wall_seg_in[0],
						'wall seg id': wall_seg_in[1],
						'wall elevation option': wall_seg_in[2],
						'elevation or height': wall_seg_in[3],
						'cylinder center x': wall_seg_in[4][0],
						'cylinder center y': wall_seg_in[4][1],
						'cylinder radius': wall_seg_in[4][2]
					}

				elif wall_seg_in[0] == "BD":
					t_seg_dict[wall_seg_in[1]] = {
						'wall type': wall_seg_in[0],
						'wall seg id': wall_seg_in[1],
						'wall elevation option': wall_seg_in[2],
						'elevation or height': wall_seg_in[3],
						'building xy points': wall_seg_in[4]
					}

			wall_segment_information_dict[wall_id] = deepcopy(t_seg_dict)
			del t_seg_dict

		# add to output dictionary/JSON
		temp_output['wall information'] = deepcopy(wall_information_dict)
		temp_output['wall segment information'] = deepcopy(wall_segment_information_dict)

		del wall_information_dict, wall_segment_information_dict

		# wall performance
		if wall_keyword == False:
			temp_output['wall performance'] = None

		elif wall_keyword == True:	
			wall_performance_dict = {}

			# add wall performance [SR, TR]
			for wall_id, perform in wall_performance.items():
				wall_performance_dict[wall_id] = {
					'speed_ratio(SR)': perform[0],
					'trap_ratio(TR)': perform[1]
				}

			temp_output['wall performance'] = deepcopy(wall_performance_dict)
			del wall_performance_dict

	# create an output json file
	with open(folder_path+exportName+'.json', 'w') as fp:
		json.dump(temp_output, fp, indent=4)

	print()
	print('completed creating summary JSON file:****************************************************')
	now = datetime.now()
	print(now.strftime("%Y/%m/%d %H:%M:%S"))


	# if everything goes well - return code to signify all completed well
	return temp_output    # no error occurred - code 0 

# overall check
def SPEC_debris_barrier_platform_v8_00(json_file_name):
	"""
	SPEC-debris incoporates multiprocessing v1.00

	Creator: Enok Cheon
	Date: 2022-03-13
	Language: Python3
	License: MIT

	perform SPEC-debris simulation for the following purpose:
		1) debris-flow propagation (with and without barrier)
		2) barrier performance against debris-flow collision

	Entrainment model options:
		'Hungr': E(s) = exp(Es*ds)
		'Er': E(s) = Es*ds

	Explanation of the json input file format for SPEC-debris are described in the user manual
	"""

	# load json file
	with open(json_file_name) as json_file:
		json_opt_input_all_data = json.load(json_file)
	
	## iterate through each json input data
	return_outputs = {}

	# iterate through each key
	for json_key_number, json_input_key in enumerate(json_opt_input_all_data.keys()):

		# iterate through each input data in a given key
		for json_data_number, json_input_dict in enumerate(json_opt_input_all_data[json_input_key]): 

			# input data number
			json_input_number = (json_key_number+1)*100 + (json_data_number+1)

			try:

				if json_input_key == "SPEC-debris-closed":
					output_dict = SPEC_debris_closed_v11_50(json_file_name, json_input_dict)

				elif json_input_key == "SPEC-debris-combined":
					output_dict = SPEC_debris_combined_v11_50(json_file_name, json_input_dict)

				elif json_input_key == "SPEC-debris":
					output_dict = SPEC_debris_v11_50(json_file_name, json_input_dict, wall_keyword=False)

				elif json_input_key == "SPEC-debris-wall":
					output_dict = SPEC_debris_v11_50(json_file_name, json_input_dict, wall_keyword=True)

				elif json_input_key == "closed":
					output_dict = opt_closed_location_v8_00(json_file_name, json_input_dict)

				elif json_input_key == "combined":
					output_dict = opt_combined_location_v8_00(json_file_name, json_input_dict)

				return_outputs[json_input_number] = deepcopy(output_dict)
				del output_dict

			except KeyboardInterrupt:
				print()
				print("user terminated early")
				print()
				return None

	json_file_name_temp = json_file_name[:-5] + '_overall_results.json'
	with open(json_file_name_temp, 'w') as fp:
		json.dump(return_outputs, fp, indent=4)

	print()
	
	# if everything goes well - return code to signify all completed well
	return 0    # no error occurred - code 0 


#################################################################################################################
## debugging 
#################################################################################################################
if __name__ == '__main__':

	print(
'''
################################################################################################
################################################################################################
SPEC-debris-barrier platform - PhD version (2022) by Dr. Enok Cheon

################################################################################################
Author:     Enok Cheon 
Date:       June 15, 2022
Name: 		SPEC-debris-barrier - PhD version
Language:   Python3
License:    

Copyright <2022> <Enok Cheon>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################################

The programming is running

'''
	)


	input_JSON_file_names = [
		'sample_input_JSON.json'
	]
	
	for input_file in input_JSON_file_names:
		return_check = check_json_input_v8_00(input_file)
		if return_check != 0 or isinstance(return_check, tuple):
			print('\n################################################################################################')
			print('return_check\n')
			print(f"error code {return_check[1]} occurred at json input number: {return_check[0]}\n")
			print(f"error title: {return_check[2]}\n")
			print(f"error title: {return_check[3]}\n")
			print('################################################################################################\n')

		else:
			return_outputs = SPEC_debris_barrier_platform_v8_00(input_file)

			print('\n################################################################################################')
			print(f'finished running the {input_file}')
			print('################################################################################################\n')

	print(
'''
################################################################################################
The programming has completed
################################################################################################
'''
	)
