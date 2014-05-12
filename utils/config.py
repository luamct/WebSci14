'''
Created on Feb 22, 2014

@author: luamct
'''

IMAGE_FOLDER     = "/scratch2/luam/images"
SCALED_FOLDER    = "/scratch2/luam/scaled"
SEGMENTED_FOLDER = "/scratch2/luam/segmented"
FEATURE_FOLDER   = "/scratch2/luam/features"
CLUSTER_FOLDER   = "/scratch2/luam/clustering"

# Database configuration

DB_HOST   = "localhost"
DB_USER   = "root"
DB_PASSWD = ""
DB_SCHEMA = "pinterest"

# Codeword generation constants

N_CODEWORDS = 4096         # Number of codewords
SAMPLE_SIZE = 1000000      # Number of descriptors sampled for clustering due to computation limitations 


MAX_PIXELS = 200000    # Maximum size in pixels of the images before that are processed


# Features extraction configurations

WAVELET_LEVELS    = 3                         # Levels considered in the Daubechies wavelet transform
COLOR_NAMES_FILE  = "color/colornames.txt"    # Path to file containing basic colors probabilities

# Mean Shift parameters

SPATIAL_RADIUS = 6
RANGE_RADIUS   = 8
MIN_DENSITY    = 1200

