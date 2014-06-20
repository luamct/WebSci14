'''
Created on Aug 12, 2013

@author: felipe costa
'''


import sys
import pylab as pp
import numpy as np
import skimage.color as sk
from sklearn.cluster import KMeans
import utils.image

import pyublas      # @UnusedImport
from sharpness._s3 import S3  # @UnresolvedImport
import pymeanshift as pms


def rgb2gray(rgb) :
	if len(rgb.shape) == 3 :
		return np.round(0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]).astype('uint8')
	else :
		return rgb


def segment(img, spatial_radius, range_radius, min_density) :
	'''
	Segment the image into regions using the Mean Shift algorithm. An image of same 
	shape is returned containing the region id of each pixel.
	'''
#	import _pymeanshift as _pms     # @UnresolvedImport

	# Check if the image has the minimum size for the algorithm to run properly 
	w, h, c = img.shape      # @UnusedVariable
	if (w <= 2*range_radius) or (h <= 2*range_radius) :
		labels_img = np.zeros(img.shape)
	
	else:
		_segmented_im, labels_img, _nregions = pms.segment(img, spatial_radius, range_radius, min_density)

	return _segmented_im, labels_img, _nregions


def normalizeLab(lab):
	""" Ranging
		L (0,100)
		a (-128,127)
		b (-128,127)
	
	"""
	lab[:,:,0] /= 100.0
	lab[:,:,1] += 128.0
	lab[:,:,2] += 128.0
	lab[:,:,1] /= 256.0
	lab[:,:,2] /= 256.0
	return lab
	
	
class Background :
		
	def __init__(self):
		self.table_name = "composition"
		
		# Load the C compiled class with fixed parameters
		self.s3 = S3(3, 2, 5, 2)

	def get_sharpness_map(self, img):
		s3map = np.zeros(img.shape, dtype=np.float64)
		self.s3.s3(img, s3map)
		return s3map


	def get_table_name(self):
		return self.table_name


	def save_mean_sharpness_map(self, rows, cols, sharp, label):

		mean_sharp_map = np.zeros((rows, cols), np.float)
		for x, y in np.ndindex((rows, cols)):
			mean_sharp_map[x,y] = sharp[label[x,y]]
			
		pp.gray()
		pp.imsave("tiger_reg_sharp.jpg", mean_sharp_map)
		
		
	def process(self, rgb, get_map=False):
		""" 
		Description data
		4 columns for each item:
		3 firsts are the Lab representation
		the last one is the sharpness's region
		"""
		result = {}
	
		gray = utils.image.rgb2gray(rgb)
		rows, cols = gray.shape
		
		# sharpness' map	
	# 	sharp = Sharpness(3, 0.8)
	# 	_result, s3map = sharp.process(imgs, get_map=True)
		s3map = self.get_sharpness_map(gray)
		
		# regions' map
		seg_im, label, reg = utils.image.segment(rgb,6,8,1200, return_colors=True)
		data = np.zeros((reg,4),dtype=np.float32)
		
		
		# get color of regions
		reg_colors = np.zeros((reg,3),dtype=np.float32)
		reg_colors[:,0] = -1
		for x, y in np.ndindex((rows, cols)):
			l = label[x,y]
			if (reg_colors[l,0] == -1) : reg_colors[l] = seg_im[x,y]
		
		# convert to lab
		temp = sk.rgb2lab(reg_colors.reshape(reg,1,3) / 255.0)
		temp = normalizeLab(temp)
		data[:,:3] = temp.reshape(reg,3)
		
		# calculate mean sharpness regions
		sharp = np.zeros((reg),dtype=np.float32)
		pix = np.zeros((reg),dtype=np.float)

		for x, y in np.ndindex((rows, cols)):
			pix[label[x,y]] += 1.0
			sharp[label[x,y]] += s3map[x,y]

		sharp /= pix # normalize
		data[:,3] = sharp

	#	aux = np.mean(data[:,3]) - np.std(data[:,3])
	#	aux = np.mean(data[:,3])
	#	aux = np.median(data[:,3])
		aux = 0.1

		for i in range(reg) :
			if data[i,3] > aux : data[i,3] += 1.0
	#		else : data[i,3] = 0.0
	#		print "["+str(data[i][0])+", "+str(data[i][1])+", "+str(data[i][2])+", "+str(data[i][3])+"],"

		# Only continue if we have at least 2 regions		
		if len(data) < 2 :
			return {'bg_area' : 0.0}

		# runs kmeans
		kmeans_model = KMeans(n_clusters=2, random_state=1).fit(data)
		clusters = kmeans_model.labels_
		
		# make map foreground
		clus_sharp = np.zeros(2)
		clus_nregs = np.zeros(2)
		clus_area  = np.zeros(2)
		for i in range(reg) :
			clus_sharp[clusters[i]] += sharp[i]
			clus_nregs[clusters[i]] += 1.0
			clus_area[clusters[i]]  += pix[i]
	
		# Calculate mean
		clus_sharp /= clus_nregs
		
		# The background is the cluster with the smaller sum of sharpness
		bg = int(clus_sharp[0] > clus_sharp[1])

	#	print clus_sharp, bg

		# Creates a visual representation with foreground=1 and background=0
		if get_map :

			visual = np.zeros((rows,cols),dtype=np.bool)
			for x, y in np.ndindex((rows, cols)):
				visual[x,y] = (clusters[label[x,y]]!=bg)
			result['bg_map'] = visual
			
# 			pp.gray()
# 			pp.imshow(visual)
# 			pp.show()

		result['bg_area'] = clus_area[bg]/ float(rows*cols)
	
	#	pp.imshow(label)
	#	pp.imshow(seg_im)
	#	pp.show()
	#	pp.imshow(s3map)	
	#	pp.show()
	#	pp.imshow(visual)

# 		pp.gray()
# 		if get_map : 
# 			pp.imsave("../images2/"+sys.argv[1]+"bj.jpg",visual)
	
	#	pp.show()
	# 	result['silhouette_score'] = silhouette_score(data, clusters, metric='euclidean')
		
		return result


def test():

	rgb  = pp.imread("../in/sorchids.jpg")
	
	bg = Background()
	x = bg.process(rgb,True)

	pp.gray()
# 	pp.imsave("tiger_bg.jpg", x['bg_map'])
	
	pp.imshow(x['bg_map'])
	pp.show()


if __name__ == '__main__' :
	test()

