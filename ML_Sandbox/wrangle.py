import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime
from dateutil import parser

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

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
pd.set_option('display.width', 1000)

FEATURE_COUNT = 0


#================================================================================
# DATA PREPARATION
#================================================================================

mount_training_data_tuple = ('mount1.csv', 'mount2.csv', 'mount3.csv', 'mount4.csv', 'mount5.csv', 'mount6.csv')
side_control_training_data_tuple = ('sc1.csv', 'sc2.csv', 'sc3.csv', 'sc4.csv', 'sc5.csv', 'sc6.csv')
general_jj_training_data_tuple = ('gj1.csv', 'gj2.csv', 'gj3.csv', 'gj4.csv', 'gj5.csv', 'gj6.csv')

def format_time(ts):
    """round microseconds to nearest 10th of a second"""

    t = pd.to_datetime(str(ts)) 
    s = t.strftime('%Y-%m-%d %H:%M:%S.%f')
    tail = s[-7:]
    f = round(float(tail), 3)
    temp = "%.2f" % f # round to 2 decimal places
    return "%s%s" % (s[:-7], temp[1:])

def print_full(x):
    """print the whole df"""
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def resolve_acc_gyro(df):
    """combine separate accelerometer and gyrocope rows into one row"""

    df.drop(df.columns[[0, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22]], axis=1, inplace=True)
    # TODO: Note that this is adding 8 hours to the time - mistakently thinks timezone is off. 
    df['timestamp'] = df['timestamp'].apply(lambda x: parser.parse(x))
    # TO DISCUSS
    # we will try merging the data based on timestamp, but we need to be forgiving to allow for small differences, 
    # so we round the microseconds
    df['timestamp'] = df['timestamp'].apply(lambda x: format_time(x))
    #df['modified_timestamp'] = df['timestamp'].shift() == df['timestamp']
    df_accel = df[df['SENSOR_TYPE'] == 'Accel_Log'].copy()

    df_accel['ACCEL_X'] = df_accel['X_AXIS']
    df_accel['ACCEL_Y'] = df_accel['Y_AXIS']
    df_accel['ACCEL_Z'] = df_accel['Z_AXIS']

    df_gyro = df[df['SENSOR_TYPE'] == 'Gyro_Log'].copy()

    df_gyro['GYRO_X'] = df_gyro['X_AXIS']
    df_gyro['GYRO_Y'] = df_gyro['Y_AXIS']
    df_gyro['GYRO_Z'] = df_gyro['Z_AXIS']

    df2 = pd.merge(df_accel, df_gyro, how='outer', on=['timestamp'])

    # having done rounding, there are some rows which do not have both accelerometer and gyro data

    df2.replace(r'\s+', np.nan, regex=True)
    df2.drop(df2.columns[[0, 1, 2, 3, 4, 9, 10, 11, 12]], axis=1, inplace=True)
    return df2


def combine_csv(directory_description):
    """concatenate multiple csv files into one pandas dataframe"""

    allFiles = glob.glob(DIR + '/'+ directory_description + '/*.csv')
    df = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        df = resolve_acc_gyro(df)
        list_.append(df)
    df = pd.concat(list_)
    complete_df = df.reset_index()
    return complete_df

def concat_data(dfList = [], *args):
    data = [x for x in dfList]
    complete_df = pd.concat(data, axis=0)
    complete_df = complete_df.reset_index()
    return complete_df

# TODO: automate checking of dfs for lopsided data (i.e. accelerometer or gyroscope started late/stopped early)
def blank_filter(df):
    before = len(df)
    df = df.dropna()
    after = len(df)

    print 'Removed {} NaN rows'.format(before-after)
    return df
    # check
    #print df.isnull().values.any()
    # check columns
    #print df.isnull().any()

def set_state(df, state):
    """set the state for training
    right now I am using:

    1 = mount
    2 = side control
    3 = general jits
    """

    if state == 'mount':
        df['state'] = 1
    elif state == 'side control':
        df['state'] = 2
    elif state =='jits':
        df['state'] = 3

    return df

def create_time_sequences(df):
    """combine rows to create time sequences"""

    # .mean() ignores NaN values
    original_x = df['ACCEL_X'].astype(float)
    original_y = df['ACCEL_Y'].astype(float)
    original_z = df['ACCEL_Z'].astype(float)

    original_gx = df['GYRO_X'].astype(float)
    original_gy = df['GYRO_Y'].astype(float)
    original_gz = df['GYRO_Z'].astype(float)

    i = original_x.index.values

    # frequency is at 25Hz, so combining 5 readings creates
    # a sequence of 5 * 0.04s = 0.20s

    idx = np.array([i, i, i, i, i]).T.flatten()[:len(original_x)]
    x = original_x.groupby(idx).mean()
    y = original_y.groupby(idx).mean()
    z = original_z.groupby(idx).mean()
    gx = original_gx.groupby(idx).mean()
    gy = original_gy.groupby(idx).mean()
    gz = original_gz.groupby(idx).mean()

    avg_df = pd.DataFrame(columns=df.columns)
    avg_df.drop(avg_df.columns[[0, 1, 5]], axis=1, inplace=True)

    avg_df['ACCEL_X'] = x
    avg_df['ACCEL_Y'] = y
    avg_df['ACCEL_Z'] = z
    avg_df['GYRO_X'] = gx
    avg_df['GYRO_Y'] = gy
    avg_df['GYRO_Z'] = gz
    
    return avg_df

def rolling_average(df):
    return pd.rolling_mean(df, window=10, center=True).mean()

def rolling_median(df):
    return pd.rolling_median(df, window=2, center=True).mean()

def create_rm_feature(df):
    features = []

    original_x = df['ACCEL_X'].astype(float)
    original_y = df['ACCEL_Y'].astype(float)
    original_z = df['ACCEL_Z'].astype(float)
    original_gx = df['GYRO_X'].astype(float)
    original_gy = df['GYRO_Y'].astype(float)
    original_gz = df['GYRO_Z'].astype(float)

    i = original_x.index.values
    idx = np.array([i, i]).T.flatten()[:len(original_x)]
    x = original_x.groupby(idx).mean()
    x.name = 'ACCEL_X'
    features.append(x)

    y = original_y.groupby(idx).mean()
    y.name = 'ACCEL_Y'
    features.append(y)

    z = original_z.groupby(idx).mean()
    z.name = 'ACCEL_Z'
    features.append(z)

    gx = original_gx.groupby(idx).mean()
    gx.name = 'GYRO_X'
    features.append(gx)

    gy = original_gy.groupby(idx).mean()
    gy.name = 'GYRO_Y'
    features.append(gy)

    gz = original_gz.groupby(idx).mean()
    gz.name = 'GYRO_Z'
    features.append(gz)
    #rolling avg x
    x_ra = df['ACCEL_X'].groupby(idx).apply(rolling_median)
    x_ra.name = 'rolling_median_x'
    features.append(x_ra)

    y_ra = df['ACCEL_Y'].groupby(idx).apply(rolling_median)
    y_ra.name = 'rolling_median_y'
    features.append(y_ra)

    z_ra = df['ACCEL_Z'].groupby(idx).apply(rolling_median)
    z_ra.name = 'rolling_median_z'
    features.append(z_ra)

    gx_ra = df['GYRO_X'].groupby(idx).apply(rolling_median)
    gx_ra.name = 'rolling_median_gx'
    features.append(gx_ra)

    gy_ra = df['GYRO_Y'].groupby(idx).apply(rolling_median)
    gy_ra.name = 'rolling_median_gy'
    features.append(gy_ra)

    gz_ra = df['GYRO_Z'].groupby(idx).apply(rolling_median)
    gz_ra.name = 'rolling_median_gz'
    features.append(gz_ra)

    data = pd.concat(features, axis=1)
    return data

def prep():
    """prepare the raw sensor data"""

    mount_training_data = combine_csv('mount_raw_data')
    mtd = set_state(mount_training_data, 'mount')
    mtd_mod = create_rm_feature(mtd)

    # repeat to give the option to train on non-sequenced data
    #mtd_mod = create_time_sequences(mtd)
    mtd_mod = set_state(mtd_mod, 'mount')

    side_control_training_data = combine_csv('side_control_raw_data')
    sctd = set_state(side_control_training_data, 'side control')
    sctd_mod = create_rm_feature(sctd)
    sctd_mod = set_state(sctd_mod, 'side control')
    
    general_jj_training_data = combine_csv('general_jj_raw_data')
    jjtd = set_state(general_jj_training_data, 'jits')
    jjtd_mod = create_rm_feature(jjtd)
    jjtd_mod = set_state(jjtd_mod, 'jits')

    specific_jj_training_data = combine_csv('specific_jj_raw_data')
    sjtd = set_state(specific_jj_training_data, 'jits')
    sjtd_mod = create_rm_feature(sjtd)
    sjtd_mod = set_state(jjtd_mod, 'jits')

    training_data = concat_data([mtd_mod, sctd_mod, jjtd_mod, sjtd_mod])

    # remove NaN

    training_data = blank_filter(training_data)
    return training_data


def prep_test(el_file):
    el_file = DIR + '/test_data/' + el_file
    df = pd.DataFrame()
    df = pd.read_csv(el_file, index_col=None, header=0)
    df = resolve_acc_gyro(df)
    df = create_rm_feature(df)
    test_data = blank_filter(df)

    return test_data


test_data1 = prep_test('test1_ysc_ocg_ymount.csv')
test_data2 = prep_test('test2_ycg_ysc.csv')
training_data = prep()
print training_data

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
    """fit and predict ML algorithms"""

    y = df_train['state'].values
    X = df_train.drop(['state', 'index'], axis=1)
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=1500, n_jobs=1,
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
    """test with mixed state data
    test 1: test1_ysc_ocg_ymount: Expect to see 2s, followed by 3s, followed by 1s
    test 2: test2_ycg_ysc: Expect to see 3s followed by 2s
    """
    y = df_train['state'].values
    X = df_train.drop(['state', 'index'], axis=1)
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=1500, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    else: 
        print "Found NaN values"

    rf.fit(X_train, y_train)
    rf_pred2 = rf.predict(test_data)
    print_full(rf_pred2)


#test_model(training_data)
#trial(training_data, test_data2)

"""
Notes

@2 reading time interval = 0.08 seconds
rf prediction: 0.93972179289
Random Forest Accuracy: 0.89 (+/- 0.09)

test 1 (very bad at detecting side control)
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 2 2 2 2 3 3 2 1 3 3 3
 3 3 2 3 3 2 2 2 2 2 1 2 2 1 1 1 1 2 2 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 1 3 3 3 2 2 2 2 2 2 2 2 3 1 1 1 3 3 3 3 3 3 2 2 2 2 2 3 1 1 1 1 3
 3 3 3 3 3 3 3 2 1 3 2 2 2 3 2 2 1 1 3 1 1 1 1 1 3 3 1 2 2 1 1 2 3 2 3 3 2
 3 3 1 1 2 2 1 2 1 1 1 1 1 1 1 1 3 3 1 1 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 1 3 1 3 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1]

 test2 (very accurate)

 [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 2 2 1 2
 2 2 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 2 2 1 1 1 2 2 2 2 2 2 2
 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 3 2
 2 2 2 2 2 2 2 2 2 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2]


@5 reading time interval = 0.2 seconds

rf prediction: 0.899581589958
Random Forest Accuracy: 0.88 (+/- 0.11)

test1 (very bad at detecting side control)
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 2 1 2 1 1 1 1 1
 3 3 3 3 3 3 2 2 2 2 1 3 3 3 2 2 1 1 1 3 3 1 2 2 3 1 1 3 2 1 3 1 1 1 1 1 1
 1 1 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1]

test2 (very accurate)
[3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 1 3 3 3 3 3 3 3 3 3 3 2 1 2 2 2 3 3 2 2 2 2 2 2 2 2 2 2 3 3 3 3
 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2]


@10 reading time interval = 0.4 seconds
rf prediction: 0.909090909091
Random Forest Accuracy: 0.88 (+/- 0.11)
BUT trial is ***RIDICULOUSLY*** more accurate

test1 (so-so)
[1 1 1 1 2 2 2 2 2 2 1 3 3 1 1 3 3 3 2 3 3 2 1 3 3 1 1 2 1 2 1 3 3 3 3 3 3
 1 1 1 1]

test2 (very good)
[3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 2 2 2 2 2 3 3 3 3 3 2 2
 2 2 2 2 2 2 2 2 2 2 2]



@15 reading time intervals = 0.6 seconds
rf prediction: 0.934426229508
Random Forest Accuracy: 0.89 (+/- 0.11)

test1 (so-so)
[1 1 1 2 2 2 2 2 3 1 1 3 2 3 3 3 3 3 1 1 3 3 3 1 1 1 1]
test2 came in perfect

@20 reading time intervals = 0.8 second
rf prediction: 0.85
Random Forest Accuracy: 0.88 (+/- 0.14)

test1 (pretty good)
[2 2 2 2 3 1 3 3 3 3 1 1 3 3 1 1]

test2: (1-2 errors)
[3 3 3 3 3 3 3 3 3 3 2 2 3 3 2 2 2 2]

@25 reading time intervals = 1 second
rf prediction: 0.928571428571
Random Forest Accuracy: 0.90 (+/- 0.08)

test1 (pretty good)
[1 2 2 2 3 3 3 1 1 3 1 1]


test2 (perfect)
[3 3 3 3 3 3 3 3 3 3 3 2 2 2]

@30 reading time intervas = 1.2 seconds
rf prediction: 0.95
Random Forest Accuracy: 0.88 (+/- 0.19)

test1 (so-so)
[1 1 1 2 3 3 3 1 1 3 1]

test2 (not good)
[3 3 3 3 3 3 2 3]


@35 reading time intervals = 1.4 seconds
rf prediction: 0.846153846154
Random Forest Accuracy: 0.89 (+/- 0.13)

test1 (pretty good)
[2 3 3 1 1 2]

test2(not good)
[3 3 3 3 3 3 2]

"""

