'''
Created on Aug 8, 2013

@author: luamct
'''
import os
import time
import numpy as np
import cPickle
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer


def stop_file_exists() :
	'''	Checks if there's a file named stop on the current directory '''
	return os.path.exists("stop")

def list_key_files(keys_folder) :
	'''	Read files containing OAuth authentication keys in the specified folder '''
	return [os.path.join(keys_folder, filename) for filename in os.listdir(keys_folder)]

def read_key_file(file_path):
	''' Read the input properties file and return its data as a dict '''

	f = open(file_path, "r")
	credentials = dict([tuple(l.strip().split("=")) for l in f.readlines()])
	f.close()
	return credentials

def create_dirs(dir_path):
	''' Creates system directory if non existing '''
	if not os.path.exists(dir_path) :
		os.makedirs(dir_path)
	return dir_path

def sleep(secs) :
	''' Sleeps but keeps watching the stop file '''

	while (not stop_file_exists()) and (secs > 0) :
		time.sleep(2)
		secs -= 2

	
