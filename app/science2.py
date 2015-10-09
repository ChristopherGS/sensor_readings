# app/science.py

import os
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pylab as pl
from yahmm import *
import random
import math

from sklearn import svm

from app.data import db, query_to_list
from app.sensors.models import Experiment, Sensor

"""
Support vector machines (SVMs) are a set of supervised learning methods 
used for classification, regression and outliers detection.
"""

_basedir = os.path.abspath(os.path.dirname(__file__))
UPLOADS = 'api/training'
UPLOAD_FOLDER = os.path.join(_basedir, UPLOADS)

filename = '50_punches_labelled_pt1_pt2_combined.csv'
non_punch_filename = 'non_punch_pt1.csv'
non_punch_filename2 = 'non_punch_pt2.csv'
mix_filename = 'mix_labelled_pt1.csv'

TRAIN_DATA = UPLOAD_FOLDER + '/punch/' + filename
TRAIN_DATA2 = UPLOAD_FOLDER + '/non_punch/' + non_punch_filename
TRAIN_DATA3 = UPLOAD_FOLDER + '/non_punch/' + non_punch_filename2

TEST_DATA = UPLOAD_FOLDER + '/labelled_test/' + mix_filename

def sql_to_pandas():
    pass

def pandas_cleanup(df):
    columns = []
    df_clean = df[['ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z', 'timestamp', 'experiment_id']]
    return df_clean


columns = ['ACCELEROMETER_X',
            'ACCELEROMETER_Y',
            'ACCELEROMETER_Z',
            'GRAVITY_X',
            'GRAVITY_Y',
            'GRAVITY_Z',
            'LINEAR_ACCELERATION_X',
            'LINEAR_ACCELERATION_Y',
            'LINEAR_ACCELERATION_Z',
            'GYROSCOPE_X',
            'GYROSCOPE_Y',
            'GYROSCOPE_Z', 
            'MAGNETIC_FIELD_X',
            'MAGNETIC_FIELD_Y',
            'MAGNETIC_FIELD_Z',
            'ORIENTATION_Z',
            'ORIENTATION_X',
            'ORIENTATION_Y',
            'Time_since_start',
            'timestamp',
            'state']

            # SHOULD REMOVE THE STATE COLUMN FOR NON TEST DATA

df_punch = pd.read_csv(TRAIN_DATA, skiprows=[0], names=columns)
df_punch2 = pd.read_csv(TRAIN_DATA2, skiprows=[0], names=columns)
df_non_punch = pd.read_csv(TRAIN_DATA3, skiprows=[0], names=columns)

df_train = pd.concat([df_punch, df_punch2, df_non_punch], ignore_index=True)
print 'train dataframe has length: {}'.format(len(df_train))

df_test = pd.read_csv(TEST_DATA, skiprows=[0], names=columns)

print 'test dataframe has length: {}'.format(len(df_test))

def my_svm(id):

    """
    As other classifiers, SVC, NuSVC and LinearSVC take as input two arrays:
    an array X of size [n_samples, n_features] holding the training samples, 
    and an array y of class labels (strings or integers), size [n_samples]:
    """

    #=============================
    #TRAINING - TODO: MOVE THIS!!!
    #=============================

    x1 = df_train['ACCELEROMETER_X'].values
    x2 = df_train['ACCELEROMETER_Y'].values
    x3 = df_train['ACCELEROMETER_Z'].values

    y = df_train['state'].values

    X = np.column_stack([x1, x2, x3])

    clf = svm.SVC()
    clf.fit(X, y)


    #=============================
    #RUN DATA AGAINST THE MODEL
    #=============================

    # Load the pandas dataframe from the DB using the id
    pandas_id = id
    print 'PANDAS ID: {}'.format(pandas_id)
    query = db.session.query(Sensor).filter(Sensor.experiment_id == pandas_id)
    df = pd.read_sql_query(query.statement, query.session.bind)


    #'SELECT * FROM sensor WHERE experiment_id == 4'
    print len(df)

    _x1 = df['ACCELEROMETER_X'].values
    _x2 = df['ACCELEROMETER_Y'].values
    _x3 = df['ACCELEROMETER_Z'].values

    _X = np.column_stack([_x1, _x2, _x3])

    my_prediction = clf.predict(_X)
    prediction_df = pd.DataFrame(my_prediction, columns=['prediction'])

    df['prediction'] = my_prediction

    print len(df)
    print len(my_prediction)

    df.to_sql('sensor', query.session.bind, flavor='sqlite', index=False, if_exists='replace')

    return df

    # 1 = punch
    # 0 = other


def non_db_svm(id):

    """
    As other classifiers, SVC, NuSVC and LinearSVC take as input two arrays:
    an array X of size [n_samples, n_features] holding the training samples, 
    and an array y of class labels (strings or integers), size [n_samples]:
    """

    #=============================
    #TRAINING - TODO: MOVE THIS!!!
    #=============================

    x1 = df_train['ACCELEROMETER_X'].values
    x2 = df_train['ACCELEROMETER_Y'].values
    x3 = df_train['ACCELEROMETER_Z'].values

    y = df_train['state'].values

    X = np.column_stack([x1, x2, x3])

    clf = svm.SVC()
    clf.fit(X, y)


    #=============================
    #TRAINING - TODO: MOVE THIS!!!
    #=============================

    # Load the pandas dataframe from the DB using the id

    _x1 = df_test['ACCELEROMETER_X'].values
    _x2 = df_test['ACCELEROMETER_Y'].values
    _x3 = df_test['ACCELEROMETER_Z'].values

    _X = np.column_stack([_x1, _x2, _x3])

    my_prediction = clf.predict(_X)

    # CHECK PREDICTION ACCURACY

    prediction_df = pd.DataFrame(my_prediction, columns=['state'])
    my_test = df_test['state']
    my_real_test = pd.DataFrame(my_test, columns=['state'])

    ne = (my_real_test != prediction_df).any(1)
    ne_stacked = (my_real_test != prediction_df).stack()
    changed = ne_stacked[ne_stacked]
    changed.index.names = ['id', 'col']

    print changed

    difference_locations = np.where(my_real_test != prediction_df)

    correct_state = my_real_test.values[difference_locations]

    predicted_state = prediction_df.values[difference_locations]

    summary_df = pd.DataFrame({'correct_state': correct_state, 'predicted_state': predicted_state}, index=changed.index)

    print summary_df

    incorrect = float(len(correct_state))
    total = float(len(my_prediction))

    print incorrect
    print total
    
    accuracy = float(incorrect/total) * 100

    print 'model accuracy is: {}%'.format(100-accuracy)


    # 1 = punch
    # 0 = other



