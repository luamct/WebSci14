'''
Created on Aug 11, 2014

@author: luamct

Some later empirical experiments showed that this metric performs 
poorly to more general cases to assess the rule of thirds.
We are currently working on improvements.
'''

import numpy as np
import utils.image 

import pyublas                 # @UnusedImport
from sharpness._s3 import S3   # @UnresolvedImport


class RuleOfThirds :

	def __init__(self):

		self.table_name = "composition"

		# Load the C compiled class with fixed parameters
		self.s3 = S3(3, 2, 5, 2)


	def get_table_name(self):
		return self.table_name


	def get_sharpness_map(self, img) :

		s3map = np.zeros(img.shape, dtype=np.float64)
		self.s3.s3(img, s3map)
		return s3map

	# Gaussian attenuation function
	def f(self, d, c) :
		return c*np.exp(-162*pow(d, 2))


	def atten_map(self, w, h):
		'''
		Builds the attenuation map that will eventually be applied 
		to the sharpness map.
		'''
		aimg = np.zeros((h, w), np.float)

		w = float(w)
		h = float(h)
	
		ax1_i = h/3.0
		ax2_i = 2.0*h/3.0

		ax3_j = w/3.0
		ax4_j = 2.0*w/3.0

		for (i,j) in np.ndindex(h,w) :
			aimg[i,j] += self.f(np.abs(i-ax1_i)/h, 0.5)
			aimg[i,j] += self.f(np.abs(i-ax2_i)/h, 0.5)
			aimg[i,j] += self.f(np.abs(j-ax3_j)/w, 0.5)
			aimg[i,j] += self.f(np.abs(j-ax4_j)/w, 0.5)

		return aimg


	def process(self, img) :
		'''
		Process the image and return an approximated assessment of its 
		accordance to the rule of thirds.
		'''
		
		img = utils.image.rgb2gray(img)
		h, w = img.shape

		att_img = self.atten_map(w, h)
		
		# Get sharpness map
		s3_img = self.get_sharpness_map(img)

		# Apply attenuation map to the sharpness map.
		rot_img  = s3_img * att_img

		# Normalized by the total sharpness
		norm_rot = np.sum(rot_img)/np.sum(s3_img)

		return {'thirds' : norm_rot}
