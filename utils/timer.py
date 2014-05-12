'''
Created on Apr 18, 2013

@author: luamct
'''

import time


total = {}
current = {}

def start(label) :
	current[label] = time.time()

def stop(label):
	if (label not in total) :
		total[label] = 0

	total[label] += time.time() - current[label]

def totals() :
	total_time = sum(total.values())

	print ""
	print "%8s : %f s\n" % ("total", total_time)
	for key, value in total.items():
		print "%8s : %.2f %%" % (key, 100.0*value/total_time)
	print ""


