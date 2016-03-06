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
from sklearn.grid_search import GridSearchCV

from utilities import (format_time, print_full, combine_csv, blank_filter, concat_data, 
    resolve_acc_gyro, convert_to_words, get_position_stats)
from feature_engineering import create_rm_feature

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
TIME_SEQUENCE_LENGTH = 25

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
    elif state =='opponent_mount':
        df['state'] = 5
    elif state =='opponent_side_control':
        df['state'] = 6
    elif state =='opponent_closed_guard':
        df['state'] = 7
    elif state == 'opponent_back_control':
        df['state'] = 8
    elif state =='non_jj':
        df['state'] = 9

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
    #5 Opponent mount
    omount_td = combine_setState_createFeatures('opponent_mount_raw_data', 'opponent_mount')
    #6 Opponent side control
    osc_td = combine_setState_createFeatures('opponent_side_control_raw_data', 'opponent_side_control')
    #7 Opponent closed guard control
    ocg_td = combine_setState_createFeatures('opponent_closed_guard_raw_data', 'opponent_closed_guard')
    #8 Opponent back control
    obc_td = combine_setState_createFeatures('opponent_back_control_raw_data', 'opponent_back_control')
    #9 "Non jiu-jitsu" motion
    nonjj_td = combine_setState_createFeatures('non_jj_raw_data', 'non_jj')

    training_data = concat_data([ymount_td, ysc_td, ycg_td, ybc_td, omount_td, osc_td, 
        ocg_td, obc_td, nonjj_td])
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
                min_samples_leaf=50, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=5000, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

    else: 
        print "Found NaN values"

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_scores = cross_validation.cross_val_score(
        rf, X, df_train.state, cv=10, scoring='accuracy')
    print 'rf prediction: {}'.format(accuracy_score(y_test, rf_pred))
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))

    # Determine feature importance
    featImp = rf.feature_importances_
    print(pd.Series(featImp, index=X.columns).sort(inplace=False,ascending=False))


def trial(df_train, test_data):
    """
    Test 1: 1s followed by 3s
    """
    y = df_train['state'].values
    X = df_train.drop(['state', 'index'], axis=1)
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=50, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=5000, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    else: 
        print "Found NaN values"

    rf.fit(X_train, y_train)
    rf_pred2 = rf.predict(test_data)
    final_prediction = convert_to_words(rf_pred2)
    print_full(final_prediction)
    get_position_stats(final_prediction)


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
    trial(training_data, test_data1)
    #trial(training_data, test_data2)
    #trial(training_data, test_data3)
    #trial(training_data, test_data4)

if __name__ == "__main__":
    start()

"""
Notes

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
Opponent Back Control: 0.0
OTHER: 0.0980392156863

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

"""





