'''
Created on Aug 7, 2013

@author: luamct
'''
import scipy
import math

import matplotlib.pyplot as pp
import numpy as np
import os

import pyublas 		  			 # @UnusedImport
import _rgb2ihls           # @UnresolvedImport


def load_image(folder, image_id):
	'''
	Load image from disk using the 0-255 range.
	'''
	path = os.path.join(folder, "%d.jpg"%image_id)
	return pp.imread(path)


def save_image(folder, image_id, image):
	'''	
	Load image from disk using the 0-255 range.
	'''
	path = os.path.join(folder, "%d.jpg"%image_id)
	pp.imsave(path, image)


def exists_and_valid(folder, image_id):
	'''	
	Check if the image file exists in the given folder.
	'''
	path = os.path.join(folder, "%d.jpg"%image_id)
	return os.path.exists(path) and (os.stat(path).st_size > 0)


def rgb2gray(rgb) :
	'''
	Converts RGB to gray scale if not already in gray scale.
	'''
	if len(rgb.shape) == 3 :
		return np.round(0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]).astype('uint8')
	else :
		return rgb


def rgb2ihls(rgb) :
	''' 
	Convert image to IHLS using a c++ compiled module.
	'''
	ihls = np.empty(rgb.shape, dtype=np.float32)
	_rgb2ihls.rgb2ihls(rgb.astype(np.float32), ihls)
	return ihls


def scale_down(img, max_pixels) :
	'''
	Scale the image down to the closest possible to max_pixels while keeping the aspect ratio
	'''
	
	# Check if needs resizing down
	w,h,_c_ = img.shape
	if (w*h > max_pixels) :

		# Finds the factor each dimension should scale down so that w*h < MAX_PIXELS
		s = math.sqrt(float(max_pixels)/(w*h))
		img = scipy.misc.imresize(img, size=s)

	return img


def segment(img, spatial_radius, range_radius, min_density, return_colors=False) :
	'''
	Segment the image into regions using the Mean Shift algorithm. An image of same 
	shape is returned containing the region id of each pixel.
	'''
	import _pymeanshift as _pms     # @UnresolvedImport

	# Check if the image has the minimum size for the algorithm to run properly 
	w, h, c = img.shape      # @UnusedVariable
	if (w <= 2*range_radius) or (h <= 2*range_radius) :
		segmented_img = np.zeros((w,h), np.int)
		labels_img = np.zeros((w,h), np.int)
		nregions = 1
		
	else:
		segmented_img, labels_img, nregions = _pms.segment(img, spatial_radius, range_radius, min_density)
	
	if return_colors :
		return segmented_img, labels_img, nregions
	else :
		return labels_img


def sample(folder):
	'''
	Generator method for sampling image files from the given folder.
	'''
	import config, random

	files = os.listdir(folder)
	random.shuffle(files)

	for fn in files:
		img_path = os.path.join(config.SCALED_FOLDER, fn)
		yield img_path


def test_segmentation() :
	import config
	from matplotlib.pyplot import imread
	
	img = imread("../aesthetics/in/leaves.jpg")
# 	segm = segment(img, config.SPATIAL_RADIUS, config.RANGE_RADIUS, config.MIN_DENSITY)
# 	segm = segment(img, config.SPATIAL_RADIUS, config.RANGE_RADIUS, 6000)
# 	print np.max(segm)
	

if __name__ == "__main__" :
	test_segmentation()
