'''
Created on Oct 18, 2013

@author: luamct
'''

from dbmanager import DBManager
import config
import numpy as np
import random
from utils import plot

from sklearn import preprocessing, tree
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.svm import SVC, SVR
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_selection.rfe import RFECV
from sklearn.cross_validation import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from scipy import stats
import matplotlib.pyplot as pp
from scipy.stats.stats import pearsonr, spearmanr
import time
from sklearn.linear_model.base import LinearRegression
from sklearn.svm.classes import LinearSVC
from collections import defaultdict , Counter
import cPickle as pickle


# Constants
PCA_VARIANCE = 0.99

# Classification
SVM_KERNEL = 'rbf'
SVM_C = 10
SVM_GAMMA = 0.01

EXTRA_TREE_PARAMS = {'criterion' : 'gini', 
										 'n_estimators' : 360,
										 'min_samples_leaf' : 4,
										 'bootstrap' : True}


# Regression
#RIDGE_ALPHA = 0.001
#LASSO_ALPHA =  0.1
#SVR_C = 100



def get_pvalues(X, y, yp, coefs) :
	X = np.array(X, np.float64)
	y = np.array(y, np.float64)
	
	sse = np.sum((yp - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
	se = np.array(np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X)))) )
	t = coefs / se
	pvalues_ = 2 * (1 - stats.t.cdf(np.abs(t), y.shape[0] - X.shape[1]))
	return pvalues_


def sample(data, repins, size=None) :
	if (not size) : 
		return data, repins

	idx = random.sample(range(len(data)), size)
	return (data[idx], repins[idx])


def grid_search(data, repins) :

	x, y = to_classification(data, repins, 0.8)

	_svm_c = SVC()  
	_svm_p = [{'kernel':['linear'], 'C':[0.1, 1, 10]},
			 			{'kernel':['rbf'], 'C':[0.1, 1, 10], 'gamma':[0.1, 0.001, 0.0001]}]

	# Random forest
	_forest_c = ExtraTreesClassifier(criterion='gini')
	_forest_p = [{
							'n_estimators' : range(200, 600, 20),
#							'n_estimators' : [200,400],
							'bootstrap' : [True, False],
							'min_samples_leaf' : [3,4,5,6,10,20]
							}]
	
	_boost_c = AdaBoostClassifier()
	_boost_p = [{'n_estimators' : range(10, 200, 10)}]


	work = {
   				'Forest'  : (_forest_c, _forest_p),
#  					'SVM' : (_svm_c, _svm_p),
#   					'AdaBoost' : (_boost_c, _boost_p)
				  }

	for name, (classf, params) in work.items() :
		
		gs = GridSearchCV(classf, params, cv=3, n_jobs=4, verbose=1)
		gs.fit(x, y)

		print name
		for p, score, _ in gs.grid_scores_ :
			print p, ':', score
	
		print
		print gs.best_params_


def run_train_test(x, y, test_size):

	xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, test_size=test_size)
	c = SVC(C=SVM_C, kernel='rbf', gamma=SVM_GAMMA)
	c.fit(xtrain, ytrain)
	strain = c.score(xtrain, ytrain)
	stest  = c.score(xtest, ytest)

	print "Train: ", strain
	print "Test : ", stest


def pca(x, n=None) :

	_pca = PCA(n)
	z = _pca.fit_transform(x)
# 	print "PCA: %d to %d dimensions" % (x.shape[1], z.shape[1])
	return z


def show_range(x) :
	_min = np.min(x, axis=0)
	_max = np.max(x, axis=0)
	print _max-_min


def to_classification(data, repins, gap) :
	''' 
	Transform the problem into a binary classif_residue problem by taking the 
	extremes of the range of examples sorted by repins and separated by gap.
	'''
	s = np.argsort(repins)
	data   = data[s,:]
	repins = repins[s]
	
	extremes = (1.0-gap)/2
	size = int(len(data)*extremes)
	
	neg = data[:size]
	pos = data[-size:]

	x = np.vstack( (neg, pos) )
	y = np.hstack( (np.zeros(size), np.ones(size)) )
	return x, y


def preprocess(data, size):
	'''
	Divide into positive and negative classes and apply PCA.
	'''
	x, y = to_classification(data, size)
	x = preprocessing.scale(x)
	x = pca(x)
	return x, y


def features_importance(concepts, data, dataset, plot_bars=True):

	# Interpret data for a classif_residue task
	x, y = to_classification(data, 0.8)
	
	# Build a forest and compute the feature importance
	forest = ExtraTreesClassifier(**EXTRA_TREE_PARAMS)
	
	forest.fit(x, y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	n_features = len(concepts)
	for f in range(n_features):
		print("%d: Feature '%s' (%f)" % (f + 1, concepts[indices[f]], importances[indices[f]]))

	# Plot the feature relevances of the forest

	if plot_bars:
		topindices = indices[:12]
		plot.bars(np.arange(len(topindices)), 
							importances[topindices], 
					  	err=std[topindices], 
					  	ticklabels=np.array(concepts)[topindices],
					  	title="Feature Relevance", 
					  	outfile='charts/%s_relevance.pdf'%dataset)

	return indices


def feature_selection(x, y) :
	
	svc = LinearSVC(C=0.01, penalty="l1", dual=False, verbose = 2)
	svc.fit(x, y)
	return svc.transform(x)


def plot_bars(x, values, err, width, title, ticklabels, outfile):

	pp.title(title)

	pp.gca().set_xticks(x)
	pp.gca().set_xticklabels(ticklabels, rotation=45, fontsize=12)

	pp.errorbar(x, values, err, fmt='.', elinewidth=1, ecolor=(0.7,0.1,0), ms=0, capthick=0.7, capsize=3)
	pp.bar(x, values, width, linewidth=1, color=(0,0.2,0.5), align='center')
	pp.ylim(0)

	pp.tight_layout()
	
	pp.savefig(outfile)
	pp.show()


def set_regularization(regr, param) :
	
	if hasattr(regr, 'alpha') :
		regr.alpha = param
	elif hasattr(regr, 'C') :
		regr.C = param
	else :
		print "Couldn't find regularization parameter on regressor ", regr


def find_alpha(x, y):

	# Standardization
	x = preprocessing.scale(x)
	x = pca(x)

	regrs = { 
						'Ridge' : Ridge(fit_intercept=False),
 				  	'Lasso' : Lasso(fit_intercept=False),
  				 	  'SVR'   : SVR(kernel='rbf')
				 }

	# Alpha for linear regressors and C for SVR
	params  = np.logspace(-6, 3, 10)
	results = {name : [] for name in regrs.keys()}

	# Test each regression model
	for name, regr in regrs.items() :

		# Test each alpha
		for p in params :

			set_regularization(regr, p) 
			scores = cv.cross_val_score(regr, x, y, 
																  scoring=make_scorer(mean_squared_error, greater_is_better=True), 
																  cv=KFold(len(y), 5, shuffle=True), 
																  n_jobs=5)

			results[name].append( (p, np.mean(scores), np.std(scores)) )

	# Print results
	for name, alpha_results in results.items() :
		best_mean = 10e8
		print "\n", name
		for alpha, mean, std in alpha_results : 
			print "%.1e : %.2f +- %.2f" % (alpha, mean, std)

			# Get the best alpha
			if mean < best_mean :
				best_alpha = alpha
				best_mean  = mean

		print "Best alpha %.1e : %.2f" % (best_alpha, best_mean)


def correlation(f1, d1, f2, d2) :
	corrs = []
# 	nfeats = len(concepts)
	for i in xrange(len(f1)) :
		for j in xrange(len(f2)) :
			corr, _pvalue = pearsonr(d1[:,i], d2[:,j])
			corrs.append( (corr, ':'.join((f1[i], f2[j]))) )
	
	corrs.sort(reverse=True)
	pos = corrs[:30]
	neg = corrs[-30:]

	print
	print "\n".join(["%.4f   %s" % (c, f) for c, f in pos])

	print
	print "\n".join(["%.4f   %s" % (c, f) for c, f in neg[::-1]])




def get_corrects(aes_data, sem_data, both_data, repins, ids, gap):

	x_aes, _y = to_classification(aes_data, repins, gap)
	x_sem, _y = to_classification(sem_data, repins, gap)
	x_both, y = to_classification(both_data, repins, gap)

	train_idx, test_idx = next(iter(StratifiedShuffleSplit(y, test_size=0.3, n_iter=1, indices=False, random_state=0)))
	
	c = ExtraTreesClassifier(**EXTRA_TREE_PARAMS)

	test_ids = ids[test_idx]
	print "Test size:", len(test_ids)

	c.fit(x_aes[train_idx], y[train_idx])
	yp = c.predict(x_aes[test_idx])
	aes_corrects = set(test_ids[(yp == y[test_idx])])

	c.fit(x_sem[train_idx], y[train_idx])
	yp = c.predict(x_sem[test_idx])
	sem_corrects = set(test_ids[(yp == y[test_idx])])

	c.fit(x_both[train_idx], y[train_idx])
	yp = c.predict(x_both[test_idx])
	both_corrects = set(test_ids[(yp == y[test_idx])])

	# Dump to file for plotting
	with open("data/corrects.p", "w") as f :
		pickle.dump((aes_corrects, sem_corrects, both_corrects), f)
	
	# Return size of test set and the correctly predicted images for each subset
	return aes_corrects, sem_corrects, both_corrects


def ensemble(aes_data, sem_data, both_data):

	# Interpret data for a classif_residue task
	aes_x, y = to_classification(aes_data, 6000)
	sem_x, y = to_classification(sem_data, 6000)
	both_x, y = to_classification(both_data, 6000)
	
	datas = {'aes':aes_x, 'sem':sem_x, 'both':both_x}

	# Build a forest and compute the feature importance
# 	forests = {'aes' : SVC(C=SVM_C, kernel='rbf'),
# 				 'sem' : SVC(C=SVM_C, kernel='rbf'),
# 				 'both': SVC(C=SVM_C, kernel='rbf')}
	
	ntrees = 201
	forests = {'aes' : ExtraTreesClassifier(**EXTRA_TREE_PARAMS),
 						 'sem' : ExtraTreesClassifier(**EXTRA_TREE_PARAMS),
 						 'both': ExtraTreesClassifier(**EXTRA_TREE_PARAMS)}

	folds = StratifiedKFold(y, 3)

	results = []
	for train, test in folds :
		
		# Will accumulate predictions from all trees
		pred = np.zeros(len(test))
		
		for s in ['aes', 'sem', 'both'] :
			data = datas[s]
			forests[s].fit(data[train], y[train])

# 			pred += forests[s].predict(data[test])

			for tree in forests[s] :
				pred += tree.predict(data[test])

		# Two or more votes mean the majority was positive
		thresold = (len(datas) * ntrees)/2
		voting = (pred >= thresold)
		truth  = (y[test]==1)

		# Save accuracy for this fold
		results.append(np.mean(truth == voting))


	print np.mean(results), np.std(results)


def to_residue(db, ids, aes, sem, vis, repins, size) :
	'''
	Transforms repins values into their corresponding residue of the expected
	repins count for the given user.
	'''
	train, test = next(iter(ShuffleSplit(len(ids), test_size=size, n_iter=1)))
	
	ids1, ids2 = ids[train], ids[test]
	repins1, repins2 = repins[train], repins[test]
	
	transformer = FollowersResiduesTransformer(db)
	transformer.fit(ids1, repins1)
	residues = transformer.transform(ids2, repins2)
	
	return ids[test], aes[test], sem[test], vis[test], residues


def classif_cv(data, repins, gap) :

	# Interpret data for a classif_residue task
	x, y = to_classification(data, repins, gap)

	# Pre-processing
	x = preprocessing.scale(x)
	x = pca(x, PCA_VARIANCE)

	folds = 5
	
#	clf = SVC(C=SVM_C, kernel='poly', degree=3)
#	clf = SVC(C=SVM_C, kernel='rbf', gamma=0.1)
#	clf = RandomForestClassifier(n_estimators=RF_ESTIMATORS)
#	clf = AdaBoostClassifier(n_estimators=BOOST_ESTIMATORS) 
	clf = ExtraTreesClassifier(**EXTRA_TREE_PARAMS)

	scores = cv.cross_val_score(clf, x, y, cv=folds, n_jobs=3)
	acc_mean, acc_std = np.mean(scores), np.std(scores)

	return acc_mean, acc_std


def classif_gaps(db, ids, aes, sem, vis, soc, repins, use_residues=False):

	if use_residues :
		ids, aes, sem, vis, target = to_residue(db, ids, aes, sem, vis, repins, 0.5)
	else :
		target = repins

	aes_acc, sem_acc, vis_acc, soc_acc = [], [], [], []
	gaps = np.arange(70, 100, 10)/100.0
	for gap in gaps :
		print gap,
		aes_acc.append(classif_cv(aes, target, gap)[0])
		sem_acc.append(classif_cv(sem, target, gap)[0])
		vis_acc.append(classif_cv(vis, target, gap)[0])
		soc_acc.append(classif_cv(soc, target, gap)[0])


	if use_residues :
		np.savetxt('data/gap_res_accs.txt', [gaps, aes_acc, sem_acc, vis_acc], fmt='%.4f')
	else :
		np.savetxt('data/gap_accs.txt', [gaps, aes_acc, sem_acc, vis_acc, soc_acc], fmt='%.4f')


def find_followers_bins(db, ids, nbins):
	
	pins  = db.get_pins_info(ids)
	npins = len(pins)

	followers = Counter([pins[pid][2] for pid in ids])
	followers = sorted(followers.items())

	bins = np.linspace(0,npins,nbins+1)
	
	i = 0
	cum = 0
	divs = []
	for f, n in followers :
		cum += n
		if (cum >= bins[i]) : 
			divs.append(f)
			i += 1
			
	return divs


class FollowersResiduesTransformer :

	def __init__(self, db, per_user=True) :
		self.per_user = per_user
		self.pins_info = db.get_pins_info()


	def get_data_per_user(self, pids, repins):
		
		repins_per_user    = defaultdict(list)
		followers_per_user = {} 
		for i, pid in enumerate(pids):
			uid, ufollowers, _bfollowers = self.pins_info[pid]
			
			repins_per_user[uid].append(repins[i])
			followers_per_user[uid] = ufollowers

		nusers = len(repins_per_user)

		mean_repins = np.empty(nusers)
		nfollowers  = np.empty(nusers)
		for i, uid in enumerate(repins_per_user.keys()) :
			mean_repins[i] = np.mean(repins_per_user[uid])
			nfollowers[i] = followers_per_user[uid]

		return (nfollowers, mean_repins)


	def get_data_per_pin(self, pids, repins):
		
		nfollowers  = [self.pins_info[pid][1] for pid in pids] 
		return (np.asarray(nfollowers), repins)


	def fit(self, pids, repins):

		if self.per_user:
			nfollowers, mean_repins = self.get_data_per_user(pids, repins)
		else :
			nfollowers, mean_repins = self.get_data_per_pin(pids, repins)

#		# Ignore zero values on either dimension because of log scaling
		nonzero = np.logical_and((nfollowers>0), (mean_repins>0))

		nfollowers  = nfollowers[nonzero]
		mean_repins = mean_repins[nonzero]

		log_mean_repins = np.log10(mean_repins)
		log_nfollowers  = np.reshape(np.log10(nfollowers), (len(nfollowers),1))

#		alpha = 0.1
		self.lr = LinearRegression(fit_intercept=True)
		self.lr.fit(log_nfollowers, log_mean_repins)
	

	def transform(self, pids, repins):

		residues = []
		for i, pid in enumerate(pids) :

			# Get board followers, since it's the closest approximation to a pin's followers 
			bfollowers = self.pins_info[pid][2]
			predicted = np.power(10, self.lr.predict(np.log10(bfollowers)))
			residues.append(repins[i] - predicted[0])

		return np.asarray(residues)


def each_feature(feats, data, repins):

	print "\nALL: ", 
	classif_cv(data, repins, 0.8)
	print ""
	
	for i in xrange(len(feats)) :
		print "%28s :" % feats[i],
		classif_cv(data[:,[i]], repins, 0.8)


def followers_groups(db, ids, data, repins, divs, gap, followers="users", use_residue=True):

	n = len(divs)
#	divs, splits = split_by_followers(db, ids, n)

	pins = db.get_pins_info(ids)
	
	idx = (1 if followers=="users" else 2)
	followers = np.asarray([pins[pid][idx] for pid in ids])

	splits = []
	for i in xrange(n-1) :
		s = np.nonzero((followers>=divs[i]) & (followers<divs[i+1]))[0]

		splits.append(s)

	# Get the size of the smallest followers groups for undersampling
#	smallest = len(splits[np.argmin(map(len, splits))])
#	print smallest
	
	accs, stds = [], []
	for i, s in enumerate(splits):
		
#		s = random.sample(s, smallest)
		print "%d .. %d (%d):\t" % (divs[i], divs[i+1], len(s)), 

		_ids = ids[s]
		_data = data[s,:]
		_repins = repins[s]

		mean, std = classif_cv(_data, _repins, gap)

		accs.append(mean)
		stds.append(std)
		
		print mean

	return accs, stds


def run_followers_groups(db, ids, aes, sem, vis, repins, followers="users", use_residues=False) :

	divs = [1, 200, 600, 1700, 4700, 13000, 1000000]
#	divs = [1, 30, 80, 220, 600, 1700, 4700, 13000, 100000]

	if use_residues :
		ids, aes, sem, vis, target = to_residue(db, ids, aes, sem, vis, repins, 0.8)
	else :
		target = repins

	gap = 0.8
	aes_acc, aes_std = followers_groups(db, ids, aes, target, divs, gap, followers, use_residues)
	sem_acc, sem_std = followers_groups(db, ids, sem, target, divs, gap, followers, use_residues)
	vis_acc, vis_std = followers_groups(db, ids, vis, target, divs, gap, followers, use_residues)

	out_data = (divs, aes_acc, aes_std, sem_acc, sem_std, vis_acc, vis_std)
	pickle.dump(out_data, open('data/followers_groups_%s.p'%followers, 'wb'))


def regression_points(db):

	nfollowers, npins, means, stds = db.get_repins_mean_and_std()

	# Ignore zero values on either dimension because of log scaling
	nonzero = np.logical_and((nfollowers>0), (means>0))

	nfollowers = nfollowers[nonzero]
	npins   = npins[nonzero]
	means   = means[nonzero]
	stds    = stds[nonzero]

	log_mean_repins = np.log10(means)
	log_nfollowers  = np.reshape(np.log10(nfollowers), (len(nfollowers),1))

	lr = LinearRegression(fit_intercept=True)
	lr.fit(log_nfollowers, log_mean_repins)

	log_pred_repins = lr.predict(log_nfollowers)
	
	np.savetxt('data/followers_x_repins.txt', 
						 [log_nfollowers, log_mean_repins, log_pred_repins], fmt='%.4f')


def spearman(feats, data, repins):
	corrs = []
	for i in xrange(len(feats)) :
		corrs.append((spearmanr(data[:,i], repins)[0], feats[i])) 

	f = open("data/rankcor.txt", 'w')
	print >> f, "\n".join(["%f %s"%(corr, feat) for (corr, feat) in sorted(corrs)])
	f.close()



def main():


	aes_filter = {
		    				'colors':['dom_colors', 'colorfulness', 'black', 'blue', 
													'brown', 'gray', 'green', 'orange', 
													'pink', 'purple', 'red', 'white', 'yellow'],
								'contrast': ['Y_contrast'],
		            'composition':['detail_level', 'bg_area', 'thirds'],
		            'sharpness':['centrality', 'density', 
														 'sharp00', 'sharp01', 'sharp02', 
														 'sharp10', 'sharp11', 'sharp12', 
														 'sharp20', 'sharp21', 'sharp22'],
		            'stats':['H_mean', 'Y_mean', 'S_mean', 
												 'H_std', 'Y_std', 'S_std'],
		            'daubechies':['H_sum', 'S_sum', 'Y_sum'],
		            'shape' : ['resolution', 'aspect_ratio']
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
							'view_portrait','view_closeupmacro','view_indoor','view_outdoor',
							'setting_citylife','setting_partylife','setting_homelife','setting_sportsrecreation',
							'setting_fooddrink','sentiment_happy','sentiment_calm','sentiment_inactive',
							'sentiment_melancholic','sentiment_unpleasant','sentiment_scary','sentiment_active',
							'sentiment_euphoric','sentiment_funny','transport_cycle','transport_car',
							'transport_truckbus','transport_rail','transport_water','transport_air']

	db = DBManager(host=config.DB_HOST, 
								 user=config.DB_USER, 
								 passwd=config.DB_PASSWD, 
								 db=config.DB_SCHEMA)

	ids, repins = db.get_repins(sample=100000)

#	classif_gaps_residues(db, ids, repins, aes_filter, sem_filter)
#	sys.exit()

	_aes_feats, aes_data = db.get_data_aesthetics(aes_filter, ids)
	_sem_feats, sem_data = db.get_data_semantics(sem_filter, ids)
	_soc_feats, soc_data = db.get_data_social(ids)
	_vis_feats, vis_data = np.hstack((_aes_feats, _sem_feats)), np.hstack((aes_data, sem_data))

	print "Examples:", len(ids)
	
#	grid_search(aes_data, repins)
#	classif_gaps(db, ids, aes_data, sem_data, vis_data, soc_data, repins, use_residues=False)

#	aes_ids, sem_ids, vis_ids = get_corrects(aes_data, sem_data, vis_data, repins, ids, 0.7)

#	regression_points(db)

#	run_followers_groups(db, ids, aes_data, sem_data, vis_data, repins, 
#											 followers="users", use_residue=True)
#	
#	run_followers_groups(db, ids, aes_data, sem_data, vis_data, repins, 
#											 followers="boards", use_residues=True)


if __name__ == '__main__':
	main()
