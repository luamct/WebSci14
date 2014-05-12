'''
Created on Aug 1, 2013

@author: luamct
'''


IMAGE_FOLDER     = "/scratch/images"
SCALED_FOLDER    = "/scratch/scaled"

# Database configuration
DB_HOST   = "localhost"
DB_USER   = "root"
DB_PASSWD = ""
DB_SCHEMA = "websci14"
DB_SOCKET = "/var/run/mysqld/mysqld.sock"


MAX_PIXELS = 200000    # Maximum size in pixels of the images before that are processed

# Features extraction configurations
WAVELET_LEVELS    = 3                         # Levels considered in the Daubechies wavelet transform
COLOR_NAMES_FILE  = "color/colornames.txt"    # Path to file containing basic colors probabilities

# Mean Shift parameters
SPATIAL_RADIUS = 6
RANGE_RADIUS   = 8
MIN_DENSITY    = 1200

