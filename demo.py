'''
Created on May 2, 2014

@author: luamct
'''
import config
from dbmanager import DBManager


if __name__ == '__main__':
		
	db = DBManager(host=config.DB_HOST, 
								 user=config.DB_USER, 
								 passwd=config.DB_PASSWD, 
								 db=config.DB_SCHEMA,
								 socket=config.DB_SOCKET)

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


	# Get the pins marked with the 'useit' flag. Basically is the pins with 
	# at least one repin and belonging to a user with at least one follower.
	ids, repins = db.get_repins()

	# Get the given features from the given ids
	aes_feats, aes_data = db.get_data_aesthetics(aes_filter, ids)
	sem_feats, sem_data = db.get_data_semantics(sem_filter, ids)
	soc_feats, soc_data = db.get_data_social(ids)

	# Do something :)
	