'''
Created on 18/09/2013

@author: Felipe Costa
'''

import numpy as np
import matplotlib.pyplot as pp
from utils.image import rgb2gray
import utils

class Contrast:
	
	def __init__(self, _sum) :
			self.sum = _sum
	
	def get_table_name(self):
		return "contrast"


	def middle(self,hist,size):
		"""
		Finds the shortest middle of size equals some percent (self.sum) mass of histogram 
		"""
		i = 0
		mid_sum = np.sum(hist[0:size])
		while (i+size <= len(hist)):

			# Window shift: remove first element and add the next one
			# Only if not in the first time
			if i :
				mid_sum = mid_sum - hist[i-1] + hist[i+size-1] 

			if (mid_sum >= self.sum):
				return True
			i += 1

		return False 


	def process(self, rgb, get_hist=False):

		img = utils.image.rgb2ihls(rgb)

# 		h_gray, _ = np.histogram(img.flatten(), bins=256, range=(0,256), density=True)
		h, _ = np.histogram(img[:,:,1].flatten(), bins=256, range=(0,256), density=True)
		
		size = 0
		solution = 0

		while(size < h.size and not(solution)):
			size += 1
			solution = self.middle(h, size)
			
		result = {'Y_contrast' : size}
		
		if get_hist :
			return h, result
		else:
			return result 


def test():
	import utils
	imgs = {'ihls' : utils.image.rgb2ihls(pp.imread("../in/cat2.jpg"))}

	ctt = Contrast(0.98)
	hist, result = ctt.process(imgs, get_hist=True)
	print result
	
	pp.plot(range(256), hist)
	pp.xlim( (0,256) )
	pp.show()

if __name__ == "__main__":
	test()

