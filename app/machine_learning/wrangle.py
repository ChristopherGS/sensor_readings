import numpy as np
import pandas as pd
import os
import six.moves.cPickle as pickle

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from flask import (abort, current_app)
from app.data import db, query_to_list
from app.sensors.models import Experiment, Sensor
from sklearn.preprocessing import PolynomialFeatures
from time import time
from scipy.stats import randint as sp_randint
from operator import itemgetter

from utilities import (format_time, print_full, combine_csv, blank_filter, concat_data, resolve_acc_gyro,
                        resolve_acc_gyro_db, convert_to_words, get_position_stats)
from feature_engineering import create_rm_feature

_basedir = os.path.abspath(os.path.dirname(__file__))
PICKLE = os.path.join(_basedir, 'pickle/training.pkl')

print PICKLE
"""
This file preps the jiu-jitsu motion data for analysis:

Step 1: Combine matching gyroscope and accelerometer rows

Step 2: Combine multiple csv files

Step 3: Label training data

Step 4: Select time interval sequence length to analyze and combine

Step 5: Create combined training file

Step 6: Algorithm explorations (inc. feature engineering)

"""

DIR = os.path.dirname(os.path.realpath(__file__))
pd.set_option('display.width', 1200)

FEATURE_COUNT = 0
TIME_SEQUENCE_LENGTH = 30
polynomial_features = PolynomialFeatures(interaction_only=False, include_bias=True, degree=3)

#================================================================================
# DATA PREPARATION
#================================================================================

def set_state(df, state):
    """set the state for training"""

    if state == 'your_mount':
        df['state'] = 1
    elif state == 'your_side_control':
        df['state'] = 2
    elif state =='your_closed_guard':
        df['state'] = 3
    elif state =='your_back_control':
        df['state'] = 4
    elif state =='opponent_mount_or_sc':
        df['state'] = 5
    elif state =='opponent_closed_guard':
        df['state'] = 6
    elif state == 'opponent_back_control':
        df['state'] = 7
    elif state =='non_jj':
        df['state'] = 8

    return df


def combine_setState_createFeatures(directory, state):
    """
    convenience method to combine three steps in one function:
    (1) combine multiple csv files, (2) set their movement state for training,
    (3) combine to create time sequences and add features
    """
    combined_data = combine_csv(directory)
    combined_data_updated = set_state(combined_data, state) # TO CHECK: probably not necessary 
    feature_training_data = create_rm_feature(combined_data_updated, TIME_SEQUENCE_LENGTH)
    ready_training_data = set_state(feature_training_data, state)
    return ready_training_data


def prep():
    """prepare the raw sensor data"""

    #1 Your mount
    ymount_td = combine_setState_createFeatures('your_mount_raw_data', 'your_mount')
    #2 Your side control
    ysc_td = combine_setState_createFeatures('your_side_control_raw_data', 'your_side_control')
    #3 Your closed guard
    ycg_td = combine_setState_createFeatures('your_closed_guard_raw_data', 'your_closed_guard')
    #4 Your back control
    ybc_td = combine_setState_createFeatures('your_back_control_raw_data', 'your_back_control')
    #5 Opponent mount or opponent side control
    omountsc_td = combine_setState_createFeatures('opponent_mount_and_opponent_side_control_raw_data', 'opponent_mount_or_sc')
    #6 Opponent closed guard
    ocg_td = combine_setState_createFeatures('opponent_closed_guard_raw_data', 'opponent_closed_guard')
    #7 Opponent back control
    obc_td = combine_setState_createFeatures('opponent_back_control_raw_data', 'opponent_back_control')
    #8 "Non jiu-jitsu" motion
    nonjj_td = combine_setState_createFeatures('non_jj_raw_data', 'non_jj')

    training_data = concat_data([ymount_td, ysc_td, ycg_td, ybc_td, omountsc_td, ocg_td, obc_td, nonjj_td])
    # remove NaN
    training_data = blank_filter(training_data)
    return training_data

def prep_test(el_file):
    el_file = DIR + '/data/test_cases/' + el_file
    df = pd.DataFrame()
    df = pd.read_csv(el_file, index_col=None, header=0)
    df = resolve_acc_gyro(df)
    df = create_rm_feature(df, TIME_SEQUENCE_LENGTH)
    test_data = blank_filter(df)

    return test_data

#================================================================================
# MACHINE LEARNING
#================================================================================

"""
Things to try:

- Adjust random forest number of trees
- Adjust data time intervals
- Adjust general jj data quantity
- Add features - not sure whether to create them before or after time sequence creation

"""

def test_model(df_train):
    """check model accuracy"""

    y = df_train['state'].values
    X = df_train.drop(['state', 'index'], axis=1)

    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=8, min_samples_split=4,
                min_weight_fraction_leaf=0.0, n_estimators=5000, n_jobs=-1,
                oob_score=False, random_state=None, verbose=2,
                warm_start=False)

        
        X = polynomial_features.fit_transform(X)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

    else: 
        print "Found NaN values"

    # Get the prediction accuracy

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_scores = cross_validation.cross_val_score(
        rf, X, df_train.state, cv=10, scoring='accuracy')
    print 'rf prediction: {}'.format(accuracy_score(y_test, rf_pred))
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))


def trial(df_train, test_data):
    """
    Test 1: 1s followed by 3s
    """
    y = df_train['state'].values
    X = df_train.drop(['state', 'index'], axis=1)
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=8, min_samples_split=4,
                min_weight_fraction_leaf=0.0, n_estimators=5000, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)

        X = polynomial_features.fit_transform(X)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

    else: 
        print "Found NaN values"

    rf.fit(X_train, y_train)
    rf_pred2 = rf.predict(test_data)
    final_prediction = convert_to_words(rf_pred2)
    print_full(final_prediction)
    get_position_stats(final_prediction)


##############
#API METHODS
##############

def api_serialize():
    training_data = prep()
    y = training_data['state'].values
    X = training_data.drop(['state', 'index'], axis=1)
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=8, min_samples_split=4,
                min_weight_fraction_leaf=0.0, n_estimators=5000, n_jobs=-1,
                oob_score=False, random_state=None, verbose=2,
                warm_start=False)

        
        X = polynomial_features.fit_transform(X)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

    else: 
        print "Found NaN values"

    rf.fit(X_train, y_train)
    joblib.dump(rf, PICKLE, compress=3) 

def api_test(experiment_id_number):
    """Prepare an experiment already uploaded to the db
    to be run against the model"""

    clean_experiment_number = int(experiment_id_number)
    try: 
        query = db.session.query(Sensor)
        df = pd.read_sql_query(query.statement, query.session.bind)
        df2 = df[df['experiment_id']==clean_experiment_number]
        df2 = resolve_acc_gyro_db(df2)
        df2 = create_rm_feature(df2, TIME_SEQUENCE_LENGTH)
        test_data = blank_filter(df2)

        # TODO: NEED TO MAKE SURE FEATURE NUMBER IS THE SAME
        #Xt = df2.drop(['state', 'index'], axis=1)
        Xt = polynomial_features.fit_transform(test_data)
        return Xt

    except Exception as e:
            current_app.logger.debug('error: {}'.format(e))
            abort(500)
    

def start():
    """Start here"""
    print "Begin analysis"

    training_data = prep()
    print 'Finished preparing training data, total length: {}'.format(len(training_data))
    print training_data

    test_data1 = prep_test('test1_ymount_ycg.csv')
    test_data2 = prep_test('test2_ysc_ymount_ybc.csv')
    test_data3 = prep_test('test3_omount_ycg_ymount_ocg_obc.csv')
    test_data4 = prep_test('test4_osc_omount_ycg.csv')

    test_model(training_data)
    #trial(training_data, test_data2)

if __name__ == "__main__":
    start()


