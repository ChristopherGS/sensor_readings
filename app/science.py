# app/science.py

import pandas as pd

def sql_to_pandas():
	pass

def pandas_cleanup(df):
	columns = []
	df_clean = df[['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'timestamp', 'experiment_id']]
	return df_clean