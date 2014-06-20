'''
Created on Sep 6, 2013

@author: luamct
'''

from _s3 import S3  # @UnresolvedImport
import pyublas      # @UnusedImport

import matplotlib.pyplot as pp
from utils.image import rgb2gray 
import mahotas
import numpy as np
import utils


class Sharpness :
	'''
	Calculates features associated to the sharpness of the image.
	'''

	def __init__(self, sections, mass_perc) :

		# Number of sections on each dimension to divide the sharpness map  
		self.nsecs = sections
		
		self.mass_perc = mass_perc

		# Load the C compiled class with fixed parameters
		self.s3 = S3(3, 2, 5, 2)

		# Table to store the features
		self.table_name = "sharpness"


	def get_table_name(self):
		return self.table_name


	def centrality(self, hist, mid, max_dist):
		cent = 0.0
		for r, v in enumerate(hist) :
			cent += v * (1 - abs(r+0.5-mid)/max_dist)**2
		return cent


	def density(self, hist, min_mass) :
		w = 1
		i = 0
		while True :
			i = 0
			while (i+w <= len(hist)) :
				s = np.sum(hist[i:i+w])
				if (s >= min_mass) :
					return w
				
				i += 1
			w += 1
	

	def binarize(self, img):

		is3map = (255*img).astype(np.uint8)
		thred = mahotas.otsu( is3map )
		return (is3map > thred)


	def process(self, img, get_map=False, binarize=True) :

		img = utils.image.rgb2gray(img)
		
		# Result map
		result = {}

		s3map = np.zeros(img.shape, dtype=np.float64)
		self.s3.s3(img, s3map)
		
		rows, cols = s3map.shape

		# Start and end of each section
		rsecs = np.linspace(0,rows,self.nsecs+1).astype(np.int)
		csecs = np.linspace(0,cols,self.nsecs+1).astype(np.int)

# 		m = np.zeros((self.nsecs, self.nsecs))
		for r, c in np.ndindex(self.nsecs, self.nsecs) :
			block = s3map[rsecs[r]:rsecs[r+1], csecs[c]:csecs[c+1]]
			result['sharp%d%d'% (r,c)] = np.mean(block)

		# Binarizes the map using Otsu filter to stress the sharp ares
		if binarize :
			s3map = self.binarize(s3map)

		s3sum = np.sum(s3map)
		
		# Only calculates centrality and density if the whole map is not 0.0
		if s3sum == 0.0 :
			result['centrality'] = 0.0
			result['density']    = 0.0

		else :
			# Calculates accumulated histogram over each dimension
			rhist = np.sum(s3map, axis=1)
			chist = np.sum(s3map, axis=0)

# 			plot_histograms(rhist, chist)

			# Calculates the centrality on each dimension
			rcent = self.centrality(rhist, 0.5*rows, 0.5*(rows-1)) / s3sum
			ccent = self.centrality(chist, 0.5*cols, 0.5*(cols-1)) / s3sum

			result['centrality'] = rcent*ccent

			# Calculates the density in each dimension
			min_mass = self.mass_perc * s3sum
			rdens = float(self.density(rhist, min_mass))/rows
			cdens = float(self.density(chist, min_mass))/cols

			result['density'] = 1.0 - rdens*cdens

		if get_map:
			return result, s3map
		else :
			return result


###############################
#     Testing methods 
############################### 

def plot_histograms(rh, ch):
# 	pp.hist(rh, 256, ec='k', fc='k')
	pp.plot(np.arange(len(rh)), rh, c='k', lw=2)
	pp.xlim(0, len(rh))
	pp.xlabel("Rows")
	pp.show()


def show_results(img, s3map, result, sections):
	
	for r in xrange(sections) :
		for c in xrange(sections) :
			print "%3.2f\t" % result['sharp%d%d'%(r,c)],
		print

	print
	print "Centrality: ", result['centrality']
	print "Density   : ", result['density']

	pp.gray()
	pp.imshow(img)
	pp.figure()
	pp.imshow(s3map)
	pp.show()


def test():

	imgs = {'gray' : rgb2gray(pp.imread("../in/tiger.jpg"))}
	secs = 3
	sharp = Sharpness(secs, 0.8)

	_result, s3map = sharp.process(imgs, get_map=True, binarize=False)
	h, w = s3map.shape

	pp.gray()

	# Draw grid
	s3map[h/3.0-1:h/3.0+1,:] = 1
	s3map[2*h/3.0-1:2*h/3.0+1,:] = 1
	s3map[:, w/3.0-1:w/3.0+1] = 1
	s3map[:, 2*w/3.0-1:2*w/3.0+1] = 1
	
	pp.imsave("tiger_s3.jpg", s3map)

# 	show_results(imgs['gray'], s3map, result, secs)


def example():
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	
	imgs = {'gray' : rgb2gray(pp.imread("../in/tiger.jpg"))}

	sharp = Sharpness(3, 0.8)
	_result, s3map = sharp.process(imgs, get_map=True, binarize=False)

	rows, cols = s3map.shape

	# Calculates accumulated histogram over each dimension
	rhist = np.sum(s3map, axis=1)
	chist = np.sum(s3map, axis=0)
	
	print len(rhist)
	print len(chist)

	pp.figure(frameon=False)
	pp.gray()
	img_ax = pp.subplot(111)
	
	img_ax.imshow(s3map)

	img_ax.set_xlim( (0,cols) )
	img_ax.set_ylim( (rows,0) )
	img_ax.set_autoscale_on(False)
	
	img_ax.get_xaxis().set_visible(False)
	img_ax.get_yaxis().set_visible(False)

	divider = make_axes_locatable(img_ax)
	histc_ax = divider.append_axes("top", size=1.2, pad=0.1, sharex=img_ax)
	histr_ax = divider.append_axes("right", size=1.2, pad=0.1, sharey=img_ax)

	histc_ax.get_xaxis().set_visible(False)
	histc_ax.get_yaxis().set_visible(False)
	histc_ax.set_autoscalex_on(False)
	
	histr_ax.get_xaxis().set_visible(False)
	histr_ax.get_yaxis().set_visible(False)
	histr_ax.set_autoscaley_on(False)

# 	width = 10
# 	binsr = np.arange(0, len(rhist) + width, width)
# 	binsc = np.arange(0, len(chist) + width, width)
	
# 	histr_ax.hist(rhist, bins=binsr, orientation='horizontal')
# 	histc_ax.hist(chist, bins=binsc)
	
	histr_ax.plot(rhist, np.arange(len(rhist)))
	histc_ax.plot(np.arange(len(chist)), chist)

# 	pp.savefig('tiger.pdf')
	pp.show()

# 	pp.imsave("salad_sharpness.jpg", s3map)
# 	show_results(imgs['gray'], s3map, result, secs)


def sample():
	import config
	import utils.image

	sections = 3
	sharp = Sharpness(sections, 0.8)
	for img_path in utils.image.sample(config.SCALED_FOLDER) :

		imgs = {'gray' : rgb2gray(pp.imread(img_path))}
		result, s3map = sharp.process(imgs, get_map=True)

		show_results(imgs['gray'], s3map, result, sections)
	
		
if __name__ == "__main__" :
# 	example()
	test()
# 	sample()
	
	
