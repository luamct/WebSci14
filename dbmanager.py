'''
Created on Sep 24, 2012

@author: luam
'''
import MySQLdb
import socket
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
import numpy as np
import random
import math


class NothingToProcessException(Exception):
	""" Raised when there are no more available pins on the DB """
	pass
	

class DBManager:
	
	# Statuses enumeration
	AVAILABLE  = 'AVAILABLE'
	PROCESSING = 'PROCESSING'
	COMPLETED  = 'COMPLETED'
	ERROR      = 'ERROR'
	IGNORE     = 'IGNORE'


	def __init__(self, host, user, passwd, db, socket, table="jobs") :

		self.db = MySQLdb.connect(host=host, 
														user=user, 
														passwd=passwd, 
														db=db,
														charset="utf8",
														init_command="SET AUTOCOMMIT=0",
														unix_socket=socket)

		# Table used for controlling the access
		self.table = table
	
#		filterwarnings('ignore', category = MySQLdb.Warning)

	
	def get_next(self, status, process_id):
		'''
		Returns one available pin to be collected
		'''
		cursor = self.db.cursor()

		# Query for next available user using FOR UPDATE to avoid race conditions
		query = "SELECT image_id FROM %s WHERE status='%s' LIMIT 1 FOR UPDATE" % (self.table, status)
		cursor.execute(query)
	
		result = cursor.fetchone()
		if (result == None) :
			raise NothingToProcessException()

		(image_id,) = result

		# Mark the account as being processed
		thread_name = "%s-%d" % (socket.gethostname(), process_id)
		query = "UPDATE %s set status='PROCESSING', process_id='%s' where image_id=%s" % (self.table, thread_name, image_id)
		cursor.execute(query)

		self.db.commit()
		cursor.close()

		return image_id

	
	def update_status(self, image_id, status) :
		''' 
		Update pin collecting status. 
		'''
		cursor = self.db.cursor()
	
		query = "UPDATE %s set status='%s' WHERE image_id=%s" % (self.table, status, image_id)
	
		cursor.execute(query)
	
		self.db.commit()
		cursor.close()


	def save_features(self, table, image_id, features) :
		'''
		Store the calculated features in the given table.
		'''
		attrs  = ",".join(features.keys()) 
		values = ",".join(map(str,features.values()))
		update = ",".join("%s=%s"%(key, str(value)) for key, value in features.items())
		query = "INSERT INTO %s (image_id, %s) VALUES (%d, %s) ON DUPLICATE KEY UPDATE %s" % (table, attrs, image_id, values, update)

		cursor = self.db.cursor()
		cursor.execute(query)
		self.db.commit()
		cursor.close()

	

	def get_repins(self, small=False, min_repins=0, sample=None) :
		c = self.db.cursor()

		query = "SELECT id, nRepins FROM pins WHERE nRepins>=%d AND useit=1" % (min_repins)
		c.execute(query)
		rows = c.fetchall()
		c.close()

		if sample : 
			rows = random.sample(rows, sample)

		ids, repins = zip(*rows)
		return (np.asarray(ids, np.int), np.asarray(repins, np.int))



	def get_features(self, table, columns, ids) :
		'''
		Returns the values from the given columns and table. The data is represented 
		as a matrix (n_elements x n_features) with the elements sorted as the argument 
		ids and the columns sorted as the argument columns.
		'''
		c = self.db.cursor()

		select = ", ".join(["t.%s"%(column) for column in columns])

		query = "SELECT p.id, %s \
							 FROM %s t JOIN pins p ON id = image_id \
							 ORDER BY nRepins, image_id" \
								% (select, table)

		c.execute(query)
		rows = c.fetchall()
		rows_map = {row[0]: row[1:] for row in rows}
		c.close()

		# Return now if there's no data
		if not rows:
			return  [], [], {}

		data = np.empty((len(ids), len(columns)), dtype=np.float64)

		for i, pin_id in enumerate(ids):
			data[i,:] = rows_map[pin_id]

		return data


	def get_boards_info(self):
		'''
		Return the number of pins and followers of each board.
		'''
		c = self.db.cursor()
		c.execute("SELECT id, nPins, nFollowers FROM boards")
		rows = c.fetchall()
		c.close()
		
		# Represent as a dict for quick access
		boards_info = {board_id: (npins, nfollowers) for (board_id, npins, nfollowers) in rows}
		return boards_info


	def get_users_categories_features(self) :
		''' 
		Calculate category entropy and percentage of uncategorized pins for each user.
		'''

		c = self.db.cursor()
		c.execute("SELECT user_id, SUM(nPins) FROM boards GROUP BY user_id")
		user_pins = c.fetchall()

		uncategorized = {}
		category_entropy = {}
		for (user_id, npins) in user_pins :

			c.execute("SELECT category, SUM(nPins) FROM boards WHERE user_id=%s GROUP BY category", user_id)
			rows = c.fetchall()

			if len(rows)==0 :
				continue

			# First we find the percentage of uncategorized content
			entropy = 0
			blank_categ = 0
			for categ, count in rows :

				if (categ == "") :
					blank_categ = count

				p = float(count)/float(npins)
				entropy -= p*math.log(p, 2)

			# Normalize entropy
			if entropy != 0:
				entropy /= math.log(len(rows),2)

			# Represent as a dict for quick access
			uncategorized[user_id] = float(blank_categ)/float(npins)
			category_entropy[user_id] = entropy 

		c.close()
		return uncategorized, category_entropy


	def get_repinned_info(self):
		
		c = self.db.cursor()
		c.execute("SELECT user_id, sum(isRepin)/COUNT(1) FROM pins GROUP BY user_id")
		rows = c.fetchall()
		c.close()

		# Represent as a dict for quick access
		return {user_id: float(repinned) for (user_id, repinned) in rows}
	

	def get_data_aesthetics(self, aes_filter, ids):
		'''
		Read the aesthetic features from the database.
		'''
	#	if (not cache_file or not os.path.exists(cache_file)) :
	
		data = []
		for table, columns in aes_filter.items() :
			table_data = self.get_features(table, columns, ids)
			data.append(table_data)
	
		features = []
		for columns in aes_filter.values() :
			features += columns 
	
		data = np.hstack(data)
		return features, data
	
	
	def get_data_semantics(self, concepts, ids):
		'''
		Read the semantic concepts from the files. 
		'''
		table_data = self.get_features("semantics", concepts, ids)
		return concepts, table_data
	
	
	def get_data_social(self, ids) :
		'''
		Read the social features from the database.
		'''
#		data = self.get_social_features(ids)
		
		# First get some aggregated values
		boards_info = self.get_boards_info()
		repinned_info = self.get_repinned_info()
		
		uncateg, categ_entropy =  self.get_users_categories_features()

		query = """SELECT p.id as pin_id, 
							 u.id as user_id, 
							 p.nComments as comments, 
							 p.category as category, 
							 p.description as description, 
							 p.isRepin as is_repin,
							 p.date as date,
							 u.gender as gender, 
							 u.nFollowers as followers, 
							 u.nFollowing as following, 
							 u.nPins as pins,
							 u.nBoards as boards,
							 (u.website != "null") as has_website, 
							 p.board_id as board_id
							 FROM pins p JOIN users u ON p.user_id = u.id"""

		# Make query, get results and represent as map {pin_id: data} for quick access
		c = self.db.cursor()
		c.execute(query)
		rows_map = {row[0]: row[1:] for row in c.fetchall()}
		c.close()

		# Store concepts as a dict per row (pin) 
		data = [] 
		for pin_id in ids:

			(user_id, ncomments, categ, desc, is_repin, date, gender, nfollowers, nfollowing, npins, nboards, has_web, board_id) = rows_map[pin_id]

			f = {}

			# Convert to string to emphasize that this feature is categorical
#			f["ncomments"] = ncomments
			f["category"] = categ
			f["description_len"] = len(desc)
			f["is_repin"] = is_repin
			f["gender"] = gender
#			f["user_followers"] = nfollowers
			f["user_following"] = nfollowing
			f["users_pins"] = npins
			f["users_boards"] = nboards
			f["has_website"] = has_web

			f["is_product"] = (1 if '$' in desc else 0)
			f["day_of_the_week"] = (date.strftime("%a") if (date) else "")

			if nfollowers == 0 : 
				nfollowers = 1

#			f["follow_ratio"] = float(nfollowing)/nfollowers

			board_pins, board_followers = boards_info[board_id]
			f["board_pins"] = board_pins            # Total pins of the board
#			f["board_followers"] = board_followers  # Total followers of the board

			f["category_entropy"] = categ_entropy[user_id]
			f["uncategorized"] = uncateg[user_id]
			f["repinned"] = repinned_info[user_id]

			data.append(f)
			
	# 	data = data[0:4,:]
	
		# Convert categorical features to numerical representation 
		vec = DictVectorizer()
		data = vec.fit_transform(data).toarray()
		return vec.get_feature_names(), data
	
		
	def close(self) :
		self.db.close()
