'''
Created on Feb 14, 2014

@author: luamct
'''

class Shape :
	'''
	'''

	def get_table_name(self):
		return "shape"
	
	def process(self, img):

		h, w, = img.shape[:2]
		return {'resolution': w*h, 'aspect_ratio': float(w)/h}
