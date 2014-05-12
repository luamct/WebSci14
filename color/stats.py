'''
Created on 13/08/2013

@author: Felipe Costa
'''
from math import sqrt, atan2, pi
import matplotlib.pyplot as pp
import numpy as np
import utils


class Statistics :
	
	def __init__(self):
			self.table_name = "stats"
											
	def get_table_name(self):
			return self.table_name

	def process(self,rgb_img):

		# Get the right color space representation
		img = utils.image.rgb2ihls(rgb_img)

		Y = img[:,:,1]
		Y_mean = np.mean(Y)
		Y_std  = np.std(Y)

		S = img[:,:,2]
		S_mean = np.mean(S)
		S_std  = np.std(S)

		# Hue mean is calculated using circular statistics
		S /= 255.0
		As = np.sum(np.cos(img[:,:,0])*S)
		Bs = np.sum(np.sin(img[:,:,0])*S)

		# Fix negatives values
		H_mean = atan2(Bs,As)
		if H_mean<0 :
			H_mean += 2*pi 

		# Circular variance
		pixels = img.shape[0]*img.shape[1]
		H_std = 1.0 - (sqrt(As**2 + Bs**2)/pixels)

		return {'H_mean': H_mean, 'H_std': H_std,
					  'Y_mean': Y_mean, 'Y_std': Y_std,
					  'S_mean': S_mean, 'S_std': S_std }


def test():
	from utils.image import rgb2ihls
	
	imgs = {'ihls' : rgb2ihls(pp.imread("../in/purple.jpg"))}
# 	ihls = np.array([[np.pi/2,100,120], 
# 									 [np.pi/2,100,120], 
# 									 [3*np.pi/2,100,120], 
# 									 [3*np.pi/2,100,120] ]).reshape(2,2,3)
	sts = Statistics()
	print sts.process(imgs)


if __name__ == "__main__":
		test()
		
		