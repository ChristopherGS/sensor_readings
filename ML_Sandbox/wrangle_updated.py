import numpy as np
import pandas as pd
import os

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from time import time
from scipy.stats import randint as sp_randint
from operator import itemgetter

from utilities import (format_time, print_full, combine_csv, blank_filter, concat_data, 
    resolve_acc_gyro, convert_to_words, get_position_stats)
from feature_engineering import create_rm_feature

import config

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
TIME_SEQUENCE_LENGTH = config.TIME_SEQUENCE_LENGTH

"""
If I just want to only get the interaction features(not x^2, then it is enough to pass interaction_only=True
and include_bias=False. 
If you want to get higher order Polynomial features(say nth degree), pass degree=n optional parameter to Polynomial Features.
"""
polynomial_features = PolynomialFeatures(interaction_only=False, include_bias=True, degree=1)

#================================================================================
# DATA PREPARATION
#================================================================================

def set_state(df, state):
    """set the state for training"""

    if state == 'your_mount':
        df['state'] = 0
    elif state == 'your_side_control':
        df['state'] = 1
    elif state =='your_closed_guard':
        df['state'] = 2
    elif state =='your_back_control':
        df['state'] = 3
    elif state =='opponent_mount_or_sc':
        df['state'] = 4
    elif state =='opponent_closed_guard':
        df['state'] = 5
    elif state == 'opponent_back_control':
        df['state'] = 6
    elif state =='non_jj':
        df['state'] = 7

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


# specify parameters and distributions to sample from
param_dist = {
              "n_estimators": [5000, 5500],
              "max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Utility function to report best scores
def report(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def test_model(df_train):
    """check model accuracy"""

    y = df_train['state'].values
    X = df_train.drop(['state', 'index'], axis=1)

    #param_grid = [
    #  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    #  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    # ]

    param_grid = {
            "n_estimators"     : [5000, 5500],
           "criterion"         : ["gini", "entropy"],
           "max_features"      : ['auto', 'sqrt', 'log2'],
           "max_depth"         : [15, 25],
           "min_samples_split" : [2, 10] ,
           "bootstrap": [True, False]}

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

    

    # run randomized search
    #n_iter_search = 3
    #random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
    #                                   n_iter=n_iter_search)

    #start = time()
    #random_search.fit(X, y)
    #print("RandomizedSearchCV took %.2f seconds for %d candidates"
    #      " parameter settings." % ((time() - start), n_iter_search))
    #report(random_search.grid_scores_)

    # Get the prediction accuracy

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_scores = cross_validation.cross_val_score(
        rf, X, df_train.state, cv=10, scoring='accuracy')
    print 'rf prediction: {}'.format(accuracy_score(y_test, rf_pred))
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))

    # run grid search
    #grid_search = GridSearchCV(rf, param_grid=param_grid)
    #start = time()
    #grid_search.fit(X, y)

    #print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
    #  % (time() - start, len(grid_search.grid_scores_)))
    #report(grid_search.grid_scores_)

    # Determine feature importance
    featImp = rf.feature_importances_
    #print(pd.Series(featImp, index=X.columns).sort(inplace=False,ascending=False))


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

    test_data = polynomial_features.fit_transform(test_data)
    rf_pred2 = rf.predict(test_data)
    print rf_pred2
    final_prediction = convert_to_words(rf_pred2)
    print_full(final_prediction)
    get_position_stats(final_prediction)
    return rf_pred2
    #print 'parameter list: {}'.format(polynomial_features.get_params())


def start():
    """Start here"""
    print "Begin analysis"

    training_data = prep()
    print 'Finished preparing training data, total length: {}'.format(len(training_data))
    print training_data

    test_data1 = prep_test('test1_ymount_ycg.csv')
    test_data2 = prep_test('test2_ysc_ymount_ybc.csv')
    test_data3 = prep_test('test3_omount_ycg_ymount_ocg_obc.csv')
    test_data4 = prep_test('GL_TEST1_CS.csv')
    test_data5 = prep_test('GL_TEST2_CS.csv')
    test_data6 = prep_test('GL_TEST3_CS_very_still.csv')
    test_data7 = prep_test('GL_TEST1_UrsWearing.csv')

    test_model(training_data)
    trial(training_data, test_data1)
    #trial(training_data, test_data2)
    #trial(training_data, test_data3)
    trial(training_data, test_data4)
    #trial(training_data, test_data5)
    trial(training_data, test_data6)
    trial(training_data, test_data7)

if __name__ == "__main__":
    start()

"""
Rambling Notes

KEY VARIABLES

1) Frequency: 25Hz vs. 50Hz (or others?)

2) Amount of time to concatenate

3) Algorithm type

4) Number of positions to attempt to detect

5) Feature engineering - number of features and their type

6) Quantity of data

7) Number of sample users for test data

8) location? Altitude effect?

9) Sensor location on the rashguard

10) Additional readings from the board (e.g. free fall) - links to features

11) Number of sensors

Data collection constants to consider (for a given training set):

- users
- Frequency
- sensor location
- sensor test
- sensor immobilization 

@2 reading time interval = 0.08 seconds
rf prediction: 0.93972179289
Random Forest Accuracy: 0.89 (+/- 0.09)

test1 @ 10 (with rolling_max features)

Your Mount: 0.112
Your Side Control: 0.016
Your Closed Guard: 0.344
Your Back Control: 0.024
Opponent Mount: 0.192
Opponent Side Control: 0.0
Opponent Closed Guard: 0.232
Opponent Back Control: 0.0
OTHER: 0.08

test 1 @ 25

Not good at distinguishing between opponent closed guard and your mount 
(if you remove opponent closed guard, detection capability goes up massively)

Also (more minor) confusion between your closed guard and opponent mount


@1500 n_estimators

rf prediction: 0.816593886463
Random Forest Accuracy: 0.81 (+/- 0.16)

Your Mount: 0.078431372549
Your Side Control: 0.0
Your Closed Guard: 0.352941176471
Your Back Control: 0.0
Opponent Mount: 0.196078431373
Opponent Side Control: 0.0
Opponent Closed Guard: 0.294117647059
Opponent Back Control: 0.0196078431373
OTHER: 0.0588235294118


@5000 n_estimators
Your Mount: 0.0196078431373
Your Side Control: 0.0196078431373
Your Closed Guard: 0.352941176471
Your Back Control: 0.0196078431373
Opponent Mount: 0.196078431373
Opponent Side Control: 0.0
Opponent Closed Guard: 0.294117647059
Opponent Back Control: 0.


@min_leaf = 5
Your Mount: 0.12
Your Side Control: 0.02
Your Closed Guard: 0.38
Your Back Control: 0.0
Opponent Mount: 0.16
Opponent Side Control: 0.0
Opponent Closed Guard: 0.22
Opponent Back Control: 0.02
OTHER: 0.08

@min_leaf = 10
Your Mount: 0.16
Your Side Control: 0.02
Your Closed Guard: 0.36
Your Back Control: 0.02
Opponent Mount: 0.18
Opponent Side Control: 0.0
Opponent Closed Guard: 0.16
Opponent Back Control: 0.0
OTHER: 0.1



===============
ADD DATA
===============
Your Mount: 0.1
Your Side Control: 0.02
Your Closed Guard: 0.36
Your Back Control: 0.02
Opponent Mount: 0.18
Opponent Side Control: 0.0
Opponent Closed Guard: 0.18
Opponent Back Control: 0.0
OTHER: 0.140
OTHER: 0.0980392156863

Your Mount: 0.0806451612903
Your Side Control: 0.0322580645161
Your Closed Guard: 0.225806451613
Your Back Control: 0.0
Opponent Mount or Opponent Side Control: 0.322580645161
Opponent Closed Guard: 0.225806451613
Opponent Back Control: 0.0
OTHER: 0.112903225806

Your Mount: 0.0645161290323
Your Side Control: 0.0322580645161
Your Closed Guard: 0.209677419355
Your Back Control: 0.0
Opponent Mount or Opponent Side Control: 0.338709677419
Opponent Closed Guard: 0.241935483871
Opponent Back Control: 0.0
OTHER: 0.112903225806

Your Mount: 0.0806451612903
Your Side Control: 0.0322580645161
Your Closed Guard: 0.338709677419
Your Back Control: 0.0
Opponent Mount or Opponent Side Control: 0.209677419355
Opponent Closed Guard: 0.177419354839
Opponent Back Control: 0.0
OTHER: 0.161290322581

Your Mount: 0.1
Your Side Control: 0.02
Your Closed Guard: 0.36
Your Back Control: 0.02
Opponent Mount or Opponent Side Control: 0.18
Opponent Closed Guard: 0.22
Opponent Back Control: 0.0
OTHER: 0.1


=====================

@min_leaf = 15
Your Mount: 0.1
Your Side Control: 0.02
Your Closed Guard: 0.36
Your Back Control: 0.0
Opponent Mount: 0.18
Opponent Side Control: 0.0
Opponent Closed Guard: 0.22
Opponent Back Control: 0.02
OTHER: 0.1

@min_leaf = 50
Your Mount: 0.08
Your Side Control: 0.0
Your Closed Guard: 0.34
Your Back Control: 0.0
Opponent Mount: 0.18
Opponent Side Control: 0.0
Opponent Closed Guard: 0.24
Opponent Back Control: 0.02
OTHER: 0.14

@ rolling_max features
rf prediction: 0.81568627451
Random Forest Accuracy: 0.76 (+/- 0.14)
Your Mount: 0.14
Your Side Control: 0.0
Your Closed Guard: 0.36
Your Back Control: 0.0
Opponent Mount: 0.18
Opponent Side Control: 0.0
Opponent Closed Guard: 0.22
Opponent Back Control: 0.02
OTHER: 0.08


@6000 n_estimators
rf prediction: 0.760784313725
Random Forest Accuracy: 0.74 (+/- 0.17)

Your Mount: 0.137254901961
Your Side Control: 0.0
Your Closed Guard: 0.352941176471
Your Back Control: 0.0196078431373
Opponent Mount: 0.196078431373
Opponent Side Control: 0.0
Opponent Closed Guard: 0.21568627451
Opponent Back Control: 0.0
OTHER: 0.078431372549


test 1 @ 30
worse than 35

@1500 n_estimators
Your Mount: 0.047619047619
Your Side Control: 0.0
Your Closed Guard: 0.309523809524
Your Back Control: 0.0238095238095
Opponent Mount: 0.238095238095
Opponent Side Control: 0.0
Opponent Closed Guard: 0.309523809524
Opponent Back Control: 0.0
OTHER: 0.0714285714286


@2000 n_estimators
Your Mount: 0.0714285714286
Your Side Control: 0.0
Your Closed Guard: 0.333333333333
Your Back Control: 0.0238095238095
Opponent Mount: 0.214285714286
Opponent Side Control: 0.0
Opponent Closed Guard: 0.285714285714
Opponent Back Control: 0.0
OTHER: 0.0714285714286

@2500 n_estimators
Your Mount: 0.119047619048
Your Side Control: 0.0
Your Closed Guard: 0.357142857143
Your Back Control: 0.0238095238095
Opponent Mount: 0.190476190476
Opponent Side Control: 0.0
Opponent Closed Guard: 0.214285714286
Opponent Back Control: 0.0
OTHER: 0.0952380952381

@5000 n_estimators
Your Mount: 0.142857142857
Your Side Control: 0.0
Your Closed Guard: 0.357142857143
Your Back Control: 0.0238095238095
Opponent Mount: 0.190476190476
Opponent Side Control: 0.0
Opponent Closed Guard: 0.190476190476
Opponent Back Control: 0.0
OTHER: 0.0952380952381

@ rolling_max
Your Mount: 0.0952380952381
Your Side Control: 0.0
Your Closed Guard: 0.357142857143
Your Back Control: 0.0238095238095
Opponent Mount: 0.190476190476
Opponent Side Control: 0.0
Opponent Closed Guard: 0.238095238095
Opponent Back Control: 0.0
OTHER: 0.0952380952381


@6000 n_estimators
Your Mount: 0.119047619048
Your Side Control: 0.0
Your Closed Guard: 0.357142857143
Your Back Control: 0.0238095238095
Opponent Mount: 0.190476190476
Opponent Side Control: 0.0
Opponent Closed Guard: 0.238095238095
Opponent Back Control: 0.0
OTHER: 0.0714285714286


@10000 n_estimators
Your Mount: 0.119047619048
Your Side Control: 0.0
Your Closed Guard: 0.333333333333
Your Back Control: 0.0238095238095
Opponent Mount: 0.214285714286
Opponent Side Control: 0.0
Opponent Closed Guard: 0.214285714286
Opponent Back Control: 0.0
OTHER: 0.0952380952381


====================
NEW DATA
====================

Your Mount: 0.121951219512
Your Side Control: 0.0
Your Closed Guard: 0.268292682927
Your Back Control: 0.0243902439024
Opponent Mount or Opponent Side Control: 0.268292682927
Opponent Closed Guard: 0.219512195122
Opponent Back Control: 0.0
OTHER: 0.0975609756098


test 1 @ 35

better

@1500 n_estimators
rf prediction: 0.807692307692
Random Forest Accuracy: 0.76 (+/- 0.15)

Your Mount: 0.0833333333333
Your Side Control: 0.0277777777778
Your Closed Guard: 0.388888888889
Your Back Control: 0.0
Opponent Mount: 0.166666666667
Opponent Side Control: 0.0
Opponent Closed Guard: 0.25
Opponent Back Control: 0.0
OTHER: 0.0833333333333


@6000 n_estimators
rf prediction: 0.785714285714
Random Forest Accuracy: 0.76 (+/- 0.16)

Your Mount: 0.111111111111
Your Side Control: 0.0277777777778
Your Closed Guard: 0.388888888889
Your Back Control: 0.0
Opponent Mount: 0.166666666667
Opponent Side Control: 0.0
Opponent Closed Guard: 0.222222222222
Opponent Back Control: 0.0
OTHER: 0.0833333333333

test 1 @ 40

worse

rf prediction: 0.81875
Random Forest Accuracy: 0.76 (+/- 0.16)

Your Mount: 0.0625
Your Side Control: 0.03125
Your Closed Guard: 0.375
Your Back Control: 0.0
Opponent Mount: 0.1875
Opponent Side Control: 0.0
Opponent Closed Guard: 0.21875
Opponent Back Control: 0.0
OTHER: 0.125









Feature engineering ideas:

- Check for pauses for positions where you are pinned?

- Check for being bridged (you can't be bridged when you are in someone's guard) - i.e. possible in one but not the other
    --> Has a bridge occured within the last x seconds?

- Consider using polynomial features: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures

- For NaNs, replace using imputation: http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values



=======
=======


TO DELETE

1) 2 polynomial

[8 8 8 1 1 1 6 6 1 1 6 6 8 1 6 8 6 2 4 5 3 5 3 3 3 5 3 3 5 3 3 3 3 5 3 5 5
 5 5 5 5]

['OTHER', 'OTHER', 'OTHER', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'OTHER', 'your_mount', 'opponent_closed_guard', 'OTHER', 'opponent_closed_guard', 'your_side_control', 'your_back_control', 'opponent_mount_or_sc', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc']
Your Mount: 0.146341463415
Your Side Control: 0.0243902439024
Your Closed Guard: 0.268292682927
Your Back Control: 0.0243902439024
Opponent Mount or Opponent Side Control: 0.268292682927
Opponent Closed Guard: 0.146341463415
Opponent Back Control: 0.0
OTHER: 0.121951219512

[8 8 7 7 7 7 4 3 8 6 1 1 6 1 1 1 1 2 2 2 2 7 2 2 2 2 1 1 3 5 5 3 3 3 5 3 3
 4 5 3 3 5 3 1 1 1 6 6 6 2 1 6 1 6 3 5 5 5 5 5 5 5 5 5 5 5 5 5 4 7 5 7 7 7
 7 7 4 7 7 2 1]
['OTHER', 'OTHER', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_back_control', 'your_closed_guard', 'OTHER', 'opponent_closed_guard', 'your_mount', 'your_mount', 'opponent_closed_guard', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'opponent_back_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_mount', 'your_mount', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_back_control', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'your_side_control', 'your_mount', 'opponent_closed_guard', 'your_mount', 'opponent_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_back_control', 'opponent_back_control', 'opponent_mount_or_sc', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_back_control', 'opponent_back_control', 'opponent_back_control', 'your_side_control', 'your_mount']
Your Mount: 0.172839506173
Your Side Control: 0.123456790123
Your Closed Guard: 0.135802469136
Your Back Control: 0.0493827160494
Opponent Mount or Opponent Side Control: 0.234567901235
Opponent Closed Guard: 0.0864197530864
Opponent Back Control: 0.16049382716
OTHER: 0.037037037037

[8 8 7 7 7 7 7 4 4 4 1 1 6 8 1 8 1 6 1 2 2 2 2 2 2 2 2 7 5 5 5 5 5 3 3 5 5
 2 1 1 1 8 1 1 6 6 8 3 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 7 7 7 5 7 7 7 7 7 4
 2 1 6 8]

['OTHER', 'OTHER', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_mount', 'your_mount', 'opponent_closed_guard', 'OTHER', 'your_mount', 'OTHER', 'your_mount', 'opponent_closed_guard', 'your_mount', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'opponent_back_control', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_side_control', 'your_mount', 'your_mount', 'your_mount', 'OTHER', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'OTHER', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_mount_or_sc', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_back_control', 'your_side_control', 'your_mount', 'opponent_closed_guard', 'OTHER']
Your Mount: 0.141025641026
Your Side Control: 0.128205128205
Your Closed Guard: 0.0384615384615
Your Back Control: 0.0641025641026
Opponent Mount or Opponent Side Control: 0.294871794872
Opponent Closed Guard: 0.0641025641026
Opponent Back Control: 0.179487179487
OTHER: 0.0897435897436



----------------------------------------------------------

2) 3 polynomial

rf prediction: 0.797720797721
Random Forest Accuracy: 0.70 (+/- 0.17)
[8 8 8 6 1 6 1 6 1 1 6 1 8 1 6 8 6 1 4 5 3 3 3 3 3 3 3 3 5 3 3 3 3 3 3 3 5
 5 5 5 5]
['OTHER', 'OTHER', 'OTHER', 'opponent_closed_guard', 'your_mount', 'opponent_closed_guard', 'your_mount', 'opponent_closed_guard', 'your_mount', 'your_mount', 'opponent_closed_guard', 'your_mount', 'OTHER', 'your_mount', 'opponent_closed_guard', 'OTHER', 'opponent_closed_guard', 'your_mount', 'your_back_control', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc']
Your Mount: 0.170731707317
Your Side Control: 0.0
Your Closed Guard: 0.365853658537
Your Back Control: 0.0243902439024
Opponent Mount or Opponent Side Control: 0.170731707317
Opponent Closed Guard: 0.146341463415
Opponent Back Control: 0.0
OTHER: 0.121951219512

[8 8 7 4 4 4 4 4 4 1 1 1 1 1 1 1 6 1 2 2 2 7 2 2 2 2 1 1 5 5 5 5 5 3 5 5 3
 4 5 5 3 3 3 1 1 1 6 6 6 2 1 6 1 6 3 5 5 5 5 5 5 5 5 5 5 5 5 5 7 5 5 7 7 7
 7 7 7 7 7 2 6]
['OTHER', 'OTHER', 'opponent_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'your_mount', 'your_side_control', 'your_side_control', 'your_side_control', 'opponent_back_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_mount', 'your_mount', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'your_back_control', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'your_side_control', 'your_mount', 'opponent_closed_guard', 'your_mount', 'opponent_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_back_control', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_side_control', 'opponent_closed_guard']
Your Mount: 0.185185185185
Your Side Control: 0.111111111111
Your Closed Guard: 0.0740740740741
Your Back Control: 0.0864197530864
Opponent Mount or Opponent Side Control: 0.296296296296
Opponent Closed Guard: 0.0864197530864
Opponent Back Control: 0.135802469136
OTHER: 0.0246913580247

[8 8 8 7 4 4 4 4 7 4 7 4 1 1 1 6 6 6 6 6 6 8 8 8 1 6 8 1 1 2 2 2 2 2 2 2 2
 2 2 2 6 3 5 3 8 3 8 8 3 3 8 3 8 8 8 3 3 8 8 3 8 3 3 3 1 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 1 1 7 3 5 5 5 5 8 8 8 8 8 8 8 8 3 5 5 5 5 5 5 5 5 5 8 5 5 8 5 5
 5 5 3 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 1 1 8 8]
['OTHER', 'OTHER', 'OTHER', 'opponent_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'opponent_back_control', 'your_back_control', 'opponent_back_control', 'your_back_control', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'OTHER', 'OTHER', 'OTHER', 'your_mount', 'opponent_closed_guard', 'OTHER', 'your_mount', 'your_mount', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'opponent_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'OTHER', 'your_closed_guard', 'OTHER', 'OTHER', 'your_closed_guard', 'your_closed_guard', 'OTHER', 'your_closed_guard', 'OTHER', 'OTHER', 'OTHER', 'your_closed_guard', 'your_closed_guard', 'OTHER', 'OTHER', 'your_closed_guard', 'OTHER', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'your_mount', 'your_mount', 'opponent_back_control', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'OTHER', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'OTHER', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_mount', 'your_mount', 'OTHER', 'OTHER']
Your Mount: 0.0814814814815
Your Side Control: 0.0814814814815
Your Closed Guard: 0.111111111111
Your Back Control: 0.0444444444444
Opponent Mount or Opponent Side Control: 0.148148148148
Opponent Closed Guard: 0.162962962963
Opponent Back Control: 0.155555555556
OTHER: 0.214814814815

[8 8 7 4 4 7 7 4 4 4 1 1 1 1 1 1 1 6 1 2 2 2 2 2 2 2 2 3 5 5 5 5 5 3 5 5 5
 2 1 1 1 8 1 1 1 6 1 5 5 5 5 5 5 5 5 5 5 5 5 3 5 5 5 4 5 7 7 7 7 7 7 7 7 7
 2 1 1 8]
['OTHER', 'OTHER', 'opponent_back_control', 'your_back_control', 'your_back_control', 'opponent_back_control', 'opponent_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'your_mount', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_side_control', 'your_mount', 'your_mount', 'your_mount', 'OTHER', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'your_mount', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_back_control', 'opponent_mount_or_sc', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_side_control', 'your_mount', 'your_mount', 'OTHER']
Your Mount: 0.217948717949
Your Side Control: 0.128205128205
Your Closed Guard: 0.0384615384615
Your Back Control: 0.0769230769231
Opponent Mount or Opponent Side Control: 0.307692307692
Opponent Closed Guard: 0.025641025641
Opponent Back Control: 0.153846153846
OTHER: 0.0512820512821


------------------------------------

rf prediction: 0.752136752137
Random Forest Accuracy: 0.70 (+/- 0.15)
rolling_max_z        0.092743
ACCEL_Z              0.091953
rolling_median_z     0.075923
rolling_min_z        0.064011
ACCEL_X              0.061452
rolling_min_x        0.053418
rolling_median_x     0.051580
rolling_max_x        0.046579
diff_x               0.044278
std_x                0.040363
ACCEL_Y              0.030593
rolling_median_y     0.028305
diff_y               0.027948
std_y                0.026728
rolling_min_y        0.024348
diff_gx              0.023778
diff_z               0.023448
rolling_max_y        0.020182
std_gz               0.019575
std_gx               0.019268
diff_gz              0.018934
std_z                0.013970
diff_gy              0.012560
std_gy               0.011418
rolling_min_gz       0.008184
rolling_max_gz       0.007593
rolling_max_gx       0.007410
rolling_min_gy       0.007125
rolling_min_gx       0.006880
GYRO_Y               0.006683
rolling_median_gy    0.006551
GYRO_X               0.005943
rolling_median_gx    0.005756
rolling_max_gy       0.005675
GYRO_Z               0.004463
rolling_median_gz    0.004381
dtype: float64
[8 8 6 6 1 1 6 6 1 6 6 6 8 1 6 8 6 1 4 5 5 3 3 3 3 3 3 3 5 3 3 3 3 3 3 3 5
 5 5 5 5]
['OTHER', 'OTHER', 'opponent_closed_guard', 'opponent_closed_guard', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'OTHER', 'your_mount', 'opponent_closed_guard', 'OTHER', 'opponent_closed_guard', 'your_mount', 'your_back_control', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc']
Your Mount: 0.121951219512
Your Side Control: 0.0
Your Closed Guard: 0.341463414634
Your Back Control: 0.0243902439024
Opponent Mount or Opponent Side Control: 0.19512195122
Opponent Closed Guard: 0.219512195122
Opponent Back Control: 0.0
OTHER: 0.0975609756098

[8 8 4 4 4 4 4 3 4 8 1 1 1 1 8 1 1 2 1 1 2 7 2 2 2 2 1 8 5 5 5 5 5 3 5 5 3
 3 5 5 3 5 3 1 1 1 1 6 6 2 1 6 1 1 3 5 5 5 5 5 5 5 5 5 5 5 5 5 4 7 5 7 7 7
 7 7 7 7 7 2 1]
['OTHER', 'OTHER', 'your_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_closed_guard', 'your_back_control', 'OTHER', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'OTHER', 'your_mount', 'your_mount', 'your_side_control', 'your_mount', 'your_mount', 'your_side_control', 'opponent_back_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_mount', 'OTHER', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'opponent_mount_or_sc', 'your_closed_guard', 'your_mount', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'your_side_control', 'your_mount', 'opponent_closed_guard', 'your_mount', 'your_mount', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_back_control', 'opponent_back_control', 'opponent_mount_or_sc', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_side_control', 'your_mount']
Your Mount: 0.20987654321
Your Side Control: 0.0987654320988
Your Closed Guard: 0.0864197530864
Your Back Control: 0.0864197530864
Opponent Mount or Opponent Side Control: 0.296296296296
Opponent Closed Guard: 0.037037037037
Opponent Back Control: 0.123456790123
OTHER: 0.0617283950617

[8 8 8 7 4 7 7 7 8 7 4 4 1 1 6 6 6 6 6 6 6 8 8 8 8 6 8 1 1 2 2 2 2 2 2 2 2
 2 2 2 6 8 3 3 3 3 8 8 8 3 3 3 8 8 8 3 3 8 8 8 8 3 3 8 1 6 6 6 6 6 6 6 6 6
 6 6 6 1 6 6 1 8 3 5 5 5 5 8 8 8 8 8 8 8 8 8 3 5 5 5 5 5 5 5 5 8 8 8 8 8 8
 5 5 3 4 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 4 1 1 8 6]
['OTHER', 'OTHER', 'OTHER', 'opponent_back_control', 'your_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'OTHER', 'opponent_back_control', 'your_back_control', 'your_back_control', 'your_mount', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'opponent_closed_guard', 'OTHER', 'your_mount', 'your_mount', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'opponent_closed_guard', 'OTHER', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'OTHER', 'OTHER', 'OTHER', 'your_closed_guard', 'your_closed_guard', 'your_closed_guard', 'OTHER', 'OTHER', 'OTHER', 'your_closed_guard', 'your_closed_guard', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'your_closed_guard', 'your_closed_guard', 'OTHER', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'opponent_closed_guard', 'your_mount', 'opponent_closed_guard', 'opponent_closed_guard', 'your_mount', 'OTHER', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'your_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_back_control', 'your_mount', 'your_mount', 'OTHER', 'opponent_closed_guard']
Your Mount: 0.0666666666667
Your Side Control: 0.0814814814815
Your Closed Guard: 0.103703703704
Your Back Control: 0.037037037037
Opponent Mount or Opponent Side Control: 0.103703703704
Opponent Closed Guard: 0.177777777778
Opponent Back Control: 0.148148148148
OTHER: 0.281481481481

[8 8 3 4 4 3 4 4 4 4 1 1 1 6 1 6 1 1 1 2 2 2 2 2 2 2 1 3 5 5 5 5 5 3 3 5 5
 2 1 1 1 4 1 1 1 6 1 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 4 7 7 5 5 7 7 7 7 7
 2 1 1 8]
['OTHER', 'OTHER', 'your_closed_guard', 'your_back_control', 'your_back_control', 'your_closed_guard', 'your_back_control', 'your_back_control', 'your_back_control', 'your_back_control', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'your_mount', 'opponent_closed_guard', 'your_mount', 'your_mount', 'your_mount', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_side_control', 'your_mount', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_closed_guard', 'your_closed_guard', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_side_control', 'your_mount', 'your_mount', 'your_mount', 'your_back_control', 'your_mount', 'your_mount', 'your_mount', 'opponent_closed_guard', 'your_mount', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'your_back_control', 'your_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_mount_or_sc', 'opponent_mount_or_sc', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'opponent_back_control', 'your_side_control', 'your_mount', 'your_mount', 'OTHER']
Your Mount: 0.217948717949
Your Side Control: 0.115384615385
Your Closed Guard: 0.0641025641026
Your Back Control: 0.115384615385
Opponent Mount or Opponent Side Control: 0.320512820513
Opponent Closed Guard: 0.0384615384615
Opponent Back Control: 0.0897435897436
OTHER: 0.0384615384615

"""





