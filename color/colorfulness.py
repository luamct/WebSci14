'''
Created on 10/09/2013

@author: Felipe Costa
'''
import pylab as pl
import numpy as np
import skimage.color as sk
import _emd     # @UnresolvedImport
import pyublas  # @UnusedImport


class Colorfulness:
	
	def __init__(self, div):
		'''
		Creates the bins positions given the number of division
		on each dimension.
		'''
		self.div = div
		self.n_bins = div ** 3	 													# numbers of bins
		self.l_bin = 256.0 / div 													# length of bin
		self.center = sk.rgb2lab( self.center() / 255.0 )	# coordinates of bins' centers
		
		self.bins_hypot = np.empty((self.n_bins), dtype=np.float64)
		self.bins_hypot.fill( 1.0/self.n_bins ) 

		# Table to insert this feature values
		self.table_name = "colors"
		

	def get_table_name(self):
		return self.table_name
	

	def center(self):
		centers = np.zeros((self.n_bins,1,3),dtype=np.float64)
		
		for i, (x, y, z) in enumerate(np.ndindex((self.div, self.div, self.div))): # for each bin
			centers[i][0] = x, y, z
		
		# Add 0.5 to shift to the center of each bin
		return (centers + 0.5) * self.l_bin


	def counter(self,img):
		'''
		Counts the occurrence of each color bin across all pixels.
		'''
		bins = np.zeros((self.div, self.div, self.div), dtype=np.float64)
		rows, cols = img.shape[:2]
		
		for x, y in np.ndindex((rows, cols)):
			r,g,b = img[x,y,:]/self.l_bin
			bins[int(r),int(g),int(b)] += 1.0 

		# Normalize
		bins /= rows*cols
		return bins

		
	def process(self,img) :
		'''
		Calculates the colorfulness of the input image using its distance to a perfectly colored image
		'''
		bins = self.counter(img)

		# Calculates the Earth's Mover Distance between the given 
		# image and a perfectly colored image.
		return {'colorfulness' : _emd.process(self.n_bins, self.bins_hypot, bins, self.center)}


def test():
	cf = Colorfulness(4)
	img = pl.imread("../in/wood_house.jpg")
	img = img.astype(pl.float32)
	print cf.process(img)
	
	
if __name__ == "__main__":
		test()
