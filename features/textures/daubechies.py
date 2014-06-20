'''
Created on Aug 7, 2013

@author: luamct
'''
import mahotas
import matplotlib.pyplot as pp
import numpy as np
import utils



class Daubechies :

	def __init__(self, levels):
		self.table_name = "daubechies"
		self.levels = levels


	def get_table_name(self):
		return self.table_name
	

	def process(self, rgb_img):

		img = utils.image.rgb2ihls(rgb_img)
		
		f = {}
		channels = {'H' : img[:,:,0], 'S' : img[:,:,1], 'Y' : img[:,:,2]}

		# For each channel
		for c, img in channels.items() :
			
			# Initiate aggregate over all levels
			f[c+"_sum"] = 0.0

			# For each wavelet level
			for l in xrange(3) :
				dimg = mahotas.daubechies(img, 'D8')

				h, w = dimg.shape
				mh, mw = h/2, w/2

				w_lh = np.mean(np.abs(dimg[:mh,mw:]))
				w_hh = np.mean(np.abs(dimg[mh:,mw:]))
				w_hl = np.mean(np.abs(dimg[mh:,:mw]))

				# Store the value in the correct name and sum the aggregate over all levels
				w = (w_lh + w_hl + w_hh)/3 
				f[c+str(l)]  = w
				f[c+"_sum"] += w

				img = img[:mh,:mw]
			
		return f

	
def test():
	img = pp.imread("in/horizontal_blur.jpg")
	daub = Daubechies(3)
	values = daub.process(img)
	print "\n".join(map(str,values.items()))
		
	
if __name__ == "__main__" :
	test()
		
		