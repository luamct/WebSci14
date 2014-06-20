'''
Created on Feb 11, 2014

@author: luamct
'''

from utils import plot
import config
from dbmanager import DBManager

import matplotlib
import matplotlib.pyplot as pp
import numpy as np
import cPickle as pickle
from scipy.stats.stats import spearmanr
import random
import os
import scipy
from matplotlib_venn import venn3_circles, venn3
from matplotlib_venn import venn3_unweighted
from matplotlib.patches import Circle

#random.seed(42)


def split_data(data, gap) :

	extremes = (1.0-gap)/2
	size = int(len(data)*extremes)
	
	bot = data[:size]
	top = data[-size:]

	return (bot, top)


def make_ylim(vmin, vmax):
	offset = 0.03*(vmax-vmin)
	return (vmin-offset, vmax+offset)
	

def boxplots(data, data_set, feature_names) :

	ylims = {'user_followers' : (0,5000),
					 'board_pins' : (0,4000),
					 'description_len' : (0, 200),
					 'user_following' : (0,4000)}
	ylogs = ['user_followers']

	print data.shape

	matplotlib.rc('font', size=18)

	bot, top = split_data(data, 0.8)
	for f in xrange(len(feature_names)) :

		fname = feature_names[f]
		fig = pp.figure(figsize=(5,5))

		bp = pp.boxplot((bot[:,f], top[:,f]), sym="o", widths=0.8, patch_artist=True)

		pp.setp(bp['boxes'], lw=1.5, color='k', facecolor='#bbbbbb')
		pp.setp(bp['medians'], lw=2, color='k')
		pp.setp(bp['whiskers'], lw=1.3, color='k', linestyle='--')
		pp.setp(bp['caps'], lw=2)
		pp.setp(bp['fliers'], color='k', markersize=3)

		pp.title(fname)
		pp.xlim(0.4,2.6)
		pp.xticks([1,2], ['Unpopular', 'Popular'])
		
		if (fname in ylims) :
			ymin, ymax = ylims[fname]
		else:
			ymin = np.min(data[:,f])
			ymax = np.max(data[:,f])

		pp.ylim(make_ylim(ymin, ymax))
		
#		if (fname in ylogs):
#			pp.yscale('log')

		pp.tight_layout(pad=0.15)
		pp.savefig("charts/boxplots/%s/%s.pdf" % (data_set, feature_names[f]))
		pp.close(fig)

# 		pp.show()
# 		break
# 		if f>4:	break
	

def plot_cdfs(db):
		
	_ids, repins = db.get_repins()
	pins_per_user = db.get_pins_per_user()

# 	print np.mean(repins), "+-", np.std(repins)

	plot.cdf(repins, title='Repins per Pin', 
									 xlabel='Repins', ylabel='P[x $\leq$ X]', 
									 xscale='log', linewidth=3,
									 xlim=(1,1000), ylim=(0.69, 1.0), 
									 outfile='charts/repins_per_pin.pdf')
	
	
	plot.cdf(pins_per_user, title='Pins per User', 
					xlabel='Pins', ylabel='P[x $\leq$ X]', 
					xscale='log', linewidth=3, 
					outfile='charts/pins_per_user.pdf')


def plot_gaps(in_file, out_file):
	
	data = np.loadtxt(in_file, float)

	nlines = len(data)-1
	legend = ['Aesthetics', 'Semantics', 'Aes+Sem', 'Social'][:nlines]
	gaps = data[0]
	plot.lines(gaps, data[1:], legend, 
 						 xlabel='Gap (%)', ylabel='Accuracy', ylim=(0.5, 0.7),
 						 outfile=out_file)


def plot_followers_x_repins(in_file, out_file):
	
	data = np.loadtxt(in_file, float)
	
	log_nfollowers  = np.power(10, data[0])
	log_mean_repins = np.power(10, data[1])
	log_pred_repins = np.power(10, data[2])
	
	d = 0.7
	limit_above = np.power(10, data[2] + d)
	limit_below = np.power(10, data[2] - d)
	
	below = []
	middle = []
	above = []
	
	d = np.power(10, 0.7)
	n = len(log_nfollowers)
	
	for i in xrange(n) :
		if (log_mean_repins[i] > limit_above[i]) : 
			above.append(i)
		elif (log_mean_repins[i] < limit_below[i]) : 
			below.append(i) 
		else:
			middle.append(i)

	pp.xlabel('Followers')
	pp.ylabel('Repin Rate ($\overline{r_u}$)')
	pp.xscale('log')
	pp.yscale('log')
	pp.xlim((np.min(log_nfollowers),np.max(log_nfollowers)))

	pp.plot(log_nfollowers[middle], log_mean_repins[middle], 'o', ms=3, color='blue')
	pp.plot(log_nfollowers[below], log_mean_repins[below], 'o', ms=3, color='blue')
	pp.plot(log_nfollowers[above], log_mean_repins[above], 'o', ms=3, color='blue')
	
	pp.plot(log_nfollowers, log_pred_repins, lw=1.2, color='red')
#	pp.plot(log_nfollowers, limit_above, lw=0.4, color='red')
#	pp.plot(log_nfollowers, limit_below, lw=0.4, color='red')

	pp.savefig(out_file)
	pp.show()


def followers_x_pins(db):
	
	nfollowers, npins, means, stds = db.get_repins_mean_and_std()

#		# Ignore zero values on either dimension because of log scaling
	nonzero = np.logical_and((nfollowers>0), (means>0))

	nfollowers = nfollowers[nonzero]
	npins   = npins[nonzero]
	means   = means[nonzero]

	pp.xlabel('Followers')
	pp.ylabel('Pins')
#	pp.xlim((0.1,10))
	
	pp.xscale('log')
	pp.yscale('log')

	pp.plot(npins, means, 'o', ms=3, color='blue')
#	pp.plot(log_nfollowers, self.lr.predict(log_nfollowers), lw=1.5, color='red')

	pp.savefig("charts/followers_x_pins.pdf")
	pp.show()

	
def plot_followers_groups(infile, outfile):
	
	divs, aes_acc, aes_std, sem_acc, sem_std, vis_acc, vis_std = pickle.load(open(infile, 'rb'))
	
	n = len(aes_acc)
	ind = np.arange(n+1)
	width = 0.28

	pp.clf()
	rects1 = pp.bar(ind[:-1],         aes_acc, width, yerr=aes_std, ecolor='r', color='#d3d3d3')
	rects2 = pp.bar(ind[:-1]+width,   sem_acc, width, yerr=sem_std, ecolor='r', color='#778899')
	rects3 = pp.bar(ind[:-1]+2*width, vis_acc, width, yerr=vis_std, ecolor='r', color='#2f4f4f')

	pp.ylim((0.45,0.8))
	pp.xlabel('Board Followers')
	pp.ylabel('Accuracy')

	pp.xticks(ind, rotation=45)
	pp.gca().set_xticklabels( divs )

	pp.legend( (rects1[0], rects2[0], rects3[0]), 
						 ('Aesthetics', 'Semantics', 'Aes+Sem'),
						  prop={'size':16}, loc="upper center")
	
	pp.tight_layout(pad=0.2)
	pp.savefig(outfile)
	pp.show()


def control_followers(db, ids, feats, data, repins, dataset):

	divs = [1, 200, 600, 1700, 4700, 13000, 1000000]

	n = len(divs)
	pins = db.get_pins_info(ids)
	
	followers = np.asarray([pins[pid][2] for pid in ids])

	groups = []
	for i in xrange(n-1) :
		g = np.nonzero((followers>=divs[i]) & (followers<divs[i+1]))[0]
		groups.append(g)

		print "%d < followers < %d (%d)\t" % (divs[i], divs[i+1], len(g))


	corrs = np.ones((len(feats), len(groups)), float)
	for i in xrange(len(feats)) :
		print "Feature:", feats[i]

		for j in xrange(len(groups)):

			_data = data[groups[j],i]
			_repins = repins[groups[j]]

			corrs[i,j] = spearmanr(_data, _repins)[1]
			print corrs[i,j],

#	pickle.dump((feats, divs, corrs), open("data/corr_%s.txt"%dataset, "wb"))


def plot_control_followers(dataset) :
	(feats, divs, corrs) = pickle.load(open("data/corr_%s.txt"%dataset, "rb"))

	print divs
	n = len(divs)-1
	for i, f in enumerate(feats):
		print f
		pp.clf()
		pp.plot(range(n), corrs[i,:])
		pp.savefig("charts/follow_x_corr/%s.pdf"%f)



def desc_len_cdf(db, ids):
	soc_data = db.get_social_features(ids)
	lengths = [row['description_len'] for row in soc_data]
#	print lengths[:100]
	
	plot.cdf(lengths, xlim=(0,200))
	

def spearman(infile, outfile) :
	
	translation = {'uncategorized': 'uncateg',
							'category=': 'no_categ',
							'gender=female': 'female',
							'gender=None' : 'no_gender',
							'category_entropy' : 'generalist',
							'category=food_drink' : 'food_drink',
							'user_following' : '#followees',
							'users_boards' : '#boards',
							'users_pins' : '#user_pins',
							'board_pins' : '#board_pins',
							}

	f = open(infile, 'r')
	corrs, feats = [], []
	for line in f :
		corr, feat = line.strip().split()
		corr = float(corr)
		if (np.abs(corr) > 0.1) :
			corrs.append(corr)
			if feat in translation:
				feat = translation[feat]
			feats.append(feat)

	f.close()

	print corrs
	plot.bars(range(len(corrs)), corrs, ticklabels=feats, 
					  width=0.8, xlim=(-0.2, len(corrs)), ylim=(-0.36,0.36), 
					  ylabel='Correlation to Repins', 
					  outfile='charts/rankcorr.pdf')


def open_image(folder, iid) :

	path = os.path.join(folder, "%d.jpg"%iid)
	if os.path.exists(path) :
		return pp.imread(path)
	else :
		return None
	

def scale_image(img, height=None, width=None):
	h, w = img.shape[:2]

	if height != None:
		size = (height, int(float(height*w)/h))
	else:
		size = (int(float(width*h)/w), width)
	
	return scipy.misc.imresize(img, size=size, interp='bicubic')
	

def image_grid(db, folder) :
	padsz = 4
	nrows = 3
	
	rows = [make_row_image(db, folder) for _r in xrange(nrows)]
	widths = [row.shape[1] for row in rows]

	wmin = np.min(widths)
	
	# White padding to go between images
	pad = 255*np.ones((padsz, wmin, 3), dtype=np.uint8)
	
	scaled_rows = [pad]
	for row in rows :
		scaled_rows.append(scale_image(row, width=wmin))
		scaled_rows.append(pad)

	grid_img = np.vstack(scaled_rows)	
	
	pp.imshow(grid_img)
	pp.imsave("charts/grid.jpg", grid_img)
	pp.show()


def odd_shape(img):
	h, w = img.shape[:2]
	if (h/w > 3) or (w/h > 3) :
		print "Odd shape!!", w, h
		return True
	return False


def make_row_image(db, folder) :

	nimages = 10
	height = 200
#	nrows = 1
	padsz = 4

	ids, repins = db.get_repins()
	
	minr = 1
	maxr = 500

	row_images = []

	sections = np.linspace(minr, maxr, nimages+1)

	for i in xrange(len(sections)-1) :
		idx = np.where(np.logical_and(repins>=sections[i], repins<sections[i+1]))[0]

		img = open_image(folder, random.choice(idx))
		while (img==None) or (img.ndim != 3) or odd_shape(img):
			img = open_image(folder, ids[random.choice(idx)])

		img = scale_image(img, height)

		row_images.append(img)
		row_images.append(255*np.ones((height, padsz, 3), dtype=np.uint8))  # Pad


	print np.mean([r.shape[0] for r in row_images]), \
				np.mean([r.shape[2] for r in row_images])
	
	return np.hstack(row_images)


def draw_venn(infile, outfile, weighted=False) :

	with open(infile, "r") as f:
		sets = pickle.load(f)

	if weighted:
		venn3_circles(sets)
		venn = venn3(sets, set_labels=('Aesthetics', 'Semantics', 'Aes+Sem'))
		for l in venn.subset_labels : 
			l.set_fontsize(14)

	else :
		venn = venn3_unweighted(sets, set_labels=('Aesthetics', 'Semantics', 'Aes+Sem'))
		for l in venn.subset_labels : 
			l.set_fontsize(14)

		ax = pp.gca()
		for (c, r) in zip(venn.centers, venn.radii):
				circle = Circle(c, r, lw=2, alpha=1, facecolor='none')
				ax.add_patch(circle)

	pp.savefig(outfile)
	pp.show()


if __name__ == '__main__':

	aes_filter = {
		    				'colors':['dom_colors', 'colorfulness', 'black', 'blue', 
													'brown', 'gray', 'green', 'orange', 
													'pink', 'purple', 'red', 'white', 'yellow'],
								'contrast': ['Y_contrast'],
		       			'faces':['nfaces', 'total_area', 'biggest_face'],
# 		        		'pad':['pleasure', 'arousal', 'dominance'],
		            'composition':['detail_level', 'bg_area'],
		            'sharpness':['centrality', 'density', 
														 'sharp00', 'sharp01', 'sharp02', 
														 'sharp10', 'sharp11', 'sharp12', 
														 'sharp20', 'sharp21', 'sharp22'],
		            'stats':['H_mean', 'Y_mean', 'S_mean', 
												 'H_std', 'Y_std', 'S_std'],
		            'daubechies':['H_sum', 'S_sum', 'Y_sum']
							 }

	sem_filter = ['timeofday_day','timeofday_night','timeofday_sunrisesunset','celestial_sun',
							'celestial_moon', 'celestial_stars','weather_clearsky','weather_overcastsky',
							'weather_cloudysky','weather_rainbow','weather_lightning','weather_fogmist',
							'weather_snowice','combustion_flames','combustion_smoke','combustion_fireworks',
							'lighting_shadow','lighting_reflection','lighting_silhouette','lighting_lenseffect',
							'scape_mountainhill','scape_desert','scape_forestpark','scape_coast','scape_rural',
							'scape_city','scape_graffiti','water_underwater','water_seaocean','water_lake',
							'water_riverstream','water_other','flora_tree','flora_plant','flora_flower',
							'flora_grass','fauna_cat','fauna_dog','fauna_horse','fauna_fish','fauna_bird',
							'fauna_insect','fauna_spider','fauna_amphibianreptile','fauna_rodent','quantity_none',
							'quantity_one','quantity_two','quantity_three','quantity_smallgroup','quantity_biggroup',
							'age_baby','age_child','age_teenager','age_adult','age_elderly','gender_male',
							'gender_female','relation_familyfriends','relation_coworkers','relation_strangers',
							'quality_noblur','quality_partialblur','quality_completeblur','quality_motionblur',
							'quality_artifacts','style_pictureinpicture','style_circularwarp','style_graycolor',
							'style_overlay','view_portrait','view_closeupmacro','view_indoor','view_outdoor',
							'setting_citylife','setting_partylife','setting_homelife','setting_sportsrecreation',
							'setting_fooddrink','sentiment_happy','sentiment_calm','sentiment_inactive',
							'sentiment_melancholic','sentiment_unpleasant','sentiment_scary','sentiment_active',
							'sentiment_euphoric','sentiment_funny','transport_cycle','transport_car',
							'transport_truckbus','transport_rail','transport_water','transport_air']
	
	db = DBManager(host=config.DB_HOST, 
								 user=config.DB_USER, 
								 passwd=config.DB_PASSWD, 
								 db=config.DB_SCHEMA)
	
#	draw_venn("data/corrects.p", "charts/venn.pdf")

#	ids, repins = db.get_repins()

#	aes_feats, aes_data = db.get_data_aesthetics(db, aes_filter, ids)
#	sem_feats, sem_data = db.get_data_semantics(db, sem_filter, ids)
#	soc_feats, soc_data = db.get_data_social(db, ids)

#	image_grid(db, "/scratch/images")
	
#	spearman('data/rankcor.txt', 'charts/rankcor.pdf')

#	boxplots(soc_data, "soc", soc_feats)
#	control_followers(db, ids, aes_feats, aes_data, repins, "aes")
#	plot_control_followers("aes")

#	plot_followers_groups('data/followers_groups_boards.p', 'charts/followers_groups_boards.pdf')
#	plot_followers_groups('data/followers_groups_users.p', 'charts/followers_groups_users.pdf')
	
#	plot_followers_x_repins('data/followers_x_repins.txt', 'charts/followers_x_repins.pdf')
	
#	plot_gaps('data/gap_accs.txt', 'charts/gap_accs.pdf')
#	plot_gaps('data/gap_acc_residues.txt', 'charts/gap_acc_residues.pdf')
	
#	plot_gaps('data/gap_acc_residues.txt', 'charts/gap_acc_residues.pdf')
#	plot_cdfs(db)
	
	
	db.close()
