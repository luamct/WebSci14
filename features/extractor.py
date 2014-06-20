#!/usr/bin/python
'''
Created on Apr 15, 2013

@author: luamct
'''
import os
import sys
import socket
import traceback
import multiprocessing as mp

import config
import utils.inout
import utils.image 
from dbmanager import DBManager, NothingToProcessException
from composition.detail_level import DetailLevel
from composition.background import Background
from color.colorfulness import Colorfulness
from textures.daubechies import Daubechies 
from color.basic_colors import BasicColors
from sharpness.sharpness import Sharpness
from color.stats import Statistics
from color.contrast import Contrast
from composition.shape import Shape
from composition.thirds import RuleOfThirds


class ExtractorManager() :
	
	def __init__(self, thread_id, extractors):
		'''
		Since this is run on the main process, it shouldn't
		open connection or file descriptors.
		'''
		self.thread_id = thread_id

		# Features to be extracted
		self.extrators  = extractors

		from warnings import filterwarnings
		filterwarnings('ignore')


	def init(self):
		''' 
		This is actually the real constructor and is run at the start of the forked process.
		''' 
		# Database connection	
		self.db = DBManager(host=config.DB_HOST, 
												user=config.DB_USER,
												passwd=config.DB_PASSWD, 
												db=config.DB_SCHEMA,
												socket=config.DB_SOCKET,
												table="jobs")

		# Create folder for scaled images if necessary
		if (not os.path.exists(config.SCALED_FOLDER)) :
			os.makedirs(config.SCALED_FOLDER)

		# Open log file
		log_folder = "log"
		if (not os.path.exists(log_folder)) :
			os.makedirs(log_folder)

		self.log_file = open("%s/%s.%d" % (log_folder, socket.gethostname(), self.thread_id), "a")

	
	def log(self, message, show=False) :
		'''
		File for multi-threading logging.
		'''
		print >> self.log_file, message
		self.log_file.flush()
		if show:
			print message


	def run(self):

		print "Starting %s." % self.thread_id

		# Should initiate already in the forked process
		self.init()

		# Process all users allocated to this thread
		while (not utils.inout.stop_file_exists()) :

			image_id = None
			try :
				image_id = self.db.get_next(DBManager.AVAILABLE, self.thread_id)

				
				# Check if there isn't already an scaled version of the image
				if (not utils.image.exists_and_valid(config.SCALED_FOLDER, image_id)) :

					# Load original sized image 
					rgb_img = utils.image.load_image(config.IMAGE_FOLDER, image_id)
	
					# Scale it down keeping the aspect ratio
					rgb_img = utils.image.scale_down(rgb_img, config.MAX_PIXELS)

					# Save a copy on disk
					utils.image.save_image(config.SCALED_FOLDER , image_id, rgb_img)

				else :
					# Load scaled down version of the image
					rgb_img = utils.image.load_image(config.SCALED_FOLDER, image_id)

				# Process all registered extractors
				for extractor in self.extrators :
					print extractor.__class__
					
					concepts = extractor.process(rgb_img)

					self.db.save_features(extractor.get_table_name(), image_id, concepts)

				# Update the pin status to DOWNLOADED
				self.db.update_status(image_id, DBManager.COMPLETED)

				# Everything went ok if got here
				print "%s: OK" % image_id

			# Nothing to collect
			except NothingToProcessException:
				self.log("Nothing to process.", show=True)
				break

			# Any other exception we log the traceback, update the DB and life goes on.
			except Exception:

				# Could not even get image id. Must halt.
				if image_id is None:
					print traceback.format_exc()
					break

				self.log("%s: ERROR\n%s" % (image_id, traceback.format_exc()), show=True)
				self.db.update_status(image_id, DBManager.ERROR)

		# Last thing before exiting thread
		self.close()


	def close(self):
		'''Clean up routine'''
		self.db.close()


def launch(process):
	''' Target method for launching the processes '''
	process.run()



if __name__ == '__main__':

	# Configure the extractors to be extracted
	extractors = [
							Daubechies(config.WAVELET_LEVELS),
							BasicColors(config.COLOR_NAMES_FILE, 0.6),
							Sharpness(3, 0.8),
							Colorfulness(5),
							Statistics(),
							DetailLevel(),
							Background(),
							Contrast(0.98),
							Shape(),
							RuleOfThirds()
							 ]

	nprocesses = 1
	if len(sys.argv) > 1:
		nprocesses = int(sys.argv[1])

	# Removes stop file
	if utils.inout.stop_file_exists() :
		os.remove("stop")

	# Doesn't use multiprocessing if nprocesses=1 for ease debugging.
	if nprocesses == 1:
		ExtractorManager(0, extractors).run()

	else:
		for tid in xrange(nprocesses) :
			proc = mp.Process(target=launch, args=(ExtractorManager(tid, extractors),))
			proc.start()

