'''
Created on Aug 8, 2013

@author: luamct
'''
from datetime import datetime
import time


def to_mysql_date(date):
    ''' Convert datetime object to mysql string format ''' 
    return datetime.strftime(from_twitter_date(date), "%Y-%m-%d %H:%M:%S") 


def from_mysql_date(date):
    ''' Convert datetime object to mysql string format ''' 
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S") 


def from_twitter_date(date_str):
    ''' Convert a date in string twitter format to a datetime object ''' 
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S +0000 %Y")

def to_timestamp(date):
    ''' Converts datetime to unix timestamp '''
    return int(time.mktime(date.timetuple()))

