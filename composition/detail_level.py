'''
Created on Aug 22, 2013

@author: luamct
'''

import numpy as np
import utils
import config


class DetailLevel():
	'''
	Quantifies the level of detail of the image as the number of regions
	produced by a segmentation algorithm. 
	'''

	def __init__(self):
		self.table_name = "composition"

	def get_table_name(self):
		return self.table_name

	def process(self, img):
		'''
		Returns the number of regions of the segmented image, which is the region with 
		the highest index + 1, since the index starts at 0.
		'''
		
		segments = utils.image.segment(img, 
																	 config.SPATIAL_RADIUS, 
																	 config.RANGE_RADIUS, 
																	 config.MIN_DENSITY)

		return {'detail_level': np.max(segments) + 1}


def test() :
	from matplotlib.pyplot import imread 
	
	imgs = {'rgb' : imread("../in/cluttered.jpg")}
	sts = DetailLevel()
	print sts.process(imgs)


if __name__ == "__main__" :
	test()
	
	
