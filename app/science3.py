# Script Name	: science3.py

import os
import pandas as pd
import numpy as np

from flask import current_app
from app.data import db, query_to_list
from app.sensors.models import Experiment, Sensor


def fetch_data(id):
	query = db.session.query(Sensor).filter(Sensor.experiment_id == id)
	df = pd.read_sql_query(query.statement, query.session.bind)
	df = df[['id', 'ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z', 'state', 'timestamp', 'experiment_id']]
	return df


# make sure you have not cleared the DB and reset the experiment numbers

def run_science():
	mount1 = fetch_data(44)
	build_features(mount1, 'difference')


def build_features(df, feature):
	if feature == 'difference':
		print "calculating difference"
		print 'df length is: {}'.format(len(df))
		# To begin, we create time sequences 
		s = df['ACCELEROMETER_X'].astype(float)
		a = s.index.values

		# TODO: seems very inelegant
		idx = np.array([a, a, a, a, a, a, a, a, a, a]).T.flatten()[:len(a)]
		s = s.groupby(idx).mean()
		print s

		"""
			###check the math
			sample_data = df.ACCELEROMETER_X[20:30].astype(float)
			sample = sample_data.as_matrix()
			print sample
			print sample.mean()
		"""
		




