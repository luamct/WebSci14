'''
Created on Aug 7, 2013

@author: luamct
'''
import numpy as np
from matplotlib.colors import ColorConverter 

colormap = {'blue': '#0000DD', 'brown': '#842121', 'gray': '#808080',
							'purple': '#800080', 'yellow': '#e5e500', 'pink': '#ff69b4',
							'black': '#000000', 'orange': '#ffa500', 'green': '#00DD00',
							'white': '#ffffff', 'red': '#DD0000'}
	

def hex_to_rgb(value):
	return tuple(int(value[i:i+2], 16) for i in range(1, 6, 2))


class BasicColors :

	def __init__(self, color_names_file, dominant_limit):
		self.table_name = "colors"

		self.basic_colors = ['black', 'blue', 'brown', 'gray', 
												'green', 'orange', 'pink', 'purple', 
												'red', 'white', 'yellow']

		# Percentage of the total pixels the dominant colors must sum up to
		self.dominant_limit = dominant_limit
		
		# Matrix of the color probabilities for each RGB bin 
		self.color_names = np.empty((32,32,32,11))

		f = open(color_names_file, "r")
		for line in f :
			# The first 3 floats are the RGB values of the center of the bin
			# and the rest are the probabilites of each color name for that bin 
			rgb, color_probs = np.split(np.fromstring(line, sep=' '), [3])
			ri,gi,bi = (rgb/8).astype(np.uint8)
			self.color_names[ri,gi,bi,:] = color_probs
			
		f.close()


	def dominant_colors(self, colors) :
		colors = np.sort(colors)[::-1]
		
		s = 0.0
		for i in xrange(colors.size) :
			s += colors[i]
			if s >= self.dominant_limit :
				return i+1

		# Should never get here				
		return colors.size
			
	
	def get_table_name(self):
		return self.table_name
	

	def process(self, img, get_color_img=False):
		'''
		Counts the basic colors on the image using an external mapping file.
		'''
		
		# Counts each color as an aggregated probability across all pixels
		colors_count = np.zeros( len(self.basic_colors) )

		if get_color_img :
			color_img = np.empty(img.shape, np.uint8)
		
		rows, cols, _chs_ = img.shape
		for r, c in np.ndindex( (rows, cols) ) :
			# Find the bin indexes of the pixel color, get the color 
			# probabilities and add them to the final count
			ri,gi,bi = img[r,c,:]/8
			colors_probs  = self.color_names[ri,gi,bi,:]
			colors_count += colors_probs
			
			if get_color_img :
				color_img[r,c,:] = hex_to_rgb(colormap[self.basic_colors[np.argmax(colors_probs)]])

		# Normalize
		colors_count /= rows*cols

		result = dict(zip(self.basic_colors, colors_count))
		result['dom_colors'] = self.dominant_colors(colors_count)

		if get_color_img :		
			return result, color_img
		else :
			return result

	
def test():
	import matplotlib.pyplot as pp

	imgs = {'rgb' : pp.imread("../in/colorful.jpg")}
	cc = BasicColors("colornames.txt", 0.6)
	values = cc.process(imgs)
	
	print values['dom_colors']
#	print "\n".join(map(str,values.items()))


def example():
	import matplotlib.pyplot as pp
	
	imgs = {'rgb' : pp.imread("../in/yellow.jpg")}
	cc = BasicColors("colornames.txt", 0.6)
	colors = cc.process(imgs)

	del colors["dom_colors"]

# 	print "\n".join(["'%s': '',"%c for c in colors.keys()])
# 	print map(str, colors.items())

	f, ax1 = pp.subplots(1,1, figsize=(6,4.5))
	
	ticks_pos = np.arange(len(colors))
	ticks_lab = colors.keys()
	ax1.bar(ticks_pos, colors.values(), 1, lw=2, color=[colormap[cn] for cn in colors.keys()])
	ax1.set_xticks(ticks_pos+0.5)
	ax1.set_xticklabels(ticks_lab, rotation=45, fontsize=8)
	ax1.set_xlim(0,11)
	ax1.set_ylabel("Image area", labelpad=10)

# 	ax2.axis('off')
# 	ax2.imshow(imgs['rgb'])

	pp.tight_layout(pad=1)
	f.savefig("yellow.pdf")
# 	pp.show()

# 	pp.imshow(img)
# 	pp.figure()
# 	
# 	pp.hist(img.flatten(), 256, range=(0,255), fc='k', ec='k')
# 	pp.show()
	
	
def histogram():
	import matplotlib.pyplot as pp
	import utils.image 
	
	img = utils.image.rgb2gray(pp.imread("../in/yellow.jpg"))
	
	pp.gray()
	pp.imshow(img)
	pp.figure()
	
	pp.hist(img.flatten(), 256, range=(0,255), fc='k', ec='k')
	pp.show()


	
def color_img():
	import matplotlib.pyplot as pp
	
	imgs = {'rgb' : pp.imread("../in/colorful.jpg")}
	cc = BasicColors("colornames.txt", 0.6)
	_colors, color_img = cc.process(imgs, get_color_img=True)

	pp.figure()
	pp.imshow(imgs['rgb'])
	
	pp.figure()
	pp.imshow(color_img)
	
	pp.show()
	
	pp.imsave("../in/cn_colorful.jpg", color_img)


if __name__ == "__main__" :
	test()

