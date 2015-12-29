# Script Name   : science3.py

import os
import pandas as pd
import numpy as np
from scipy import stats, integrate

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

from flask import current_app
from app.data import db, query_to_list
from app.sensors.models import Experiment, Sensor




MOUNT_ID = [44, 45, 46, 50]

SIDE_CONTROL = [47, 48, 49]

GENERAL_JITS = [51, 52, 53]





def fetch_data(id):
    query = db.session.query(Sensor).filter(Sensor.experiment_id == id)
    df = pd.read_sql_query(query.statement, query.session.bind)
    df = df[['id', 'ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z', 'state', 'timestamp', 'experiment_id']]
    return df

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def concat_data(dfList = [], *args):
    data = [x for x in dfList]
    complete_df = pd.concat(data, axis=0)
    complete_df = complete_df.reset_index()
    return complete_df
    # current_app.logger.debug('Here is the concatenated df: {}'.format(print_full(complete_df)))


def set_to_int(df):
    df['ACCELEROMETER_X'] = df['ACCELEROMETER_X'].astype(int)
    return df

    
def prep_data():
     # make sure you have not cleared the DB and reset the experiment numbers
    mount1 = fetch_data(MOUNT_ID[0])
    mount2 = fetch_data(MOUNT_ID[1])
    mount3 = fetch_data(MOUNT_ID[2])
    mount4 = fetch_data(MOUNT_ID[3])

    sc1 = fetch_data(SIDE_CONTROL[0])
    sc2 = fetch_data(SIDE_CONTROL[1])
    sc3 = fetch_data(SIDE_CONTROL[2])

    general_jits1 = fetch_data(GENERAL_JITS[0])
    general_jits2 = fetch_data(GENERAL_JITS[1])
    general_jits3 = fetch_data(GENERAL_JITS[2])

    mount_df = concat_data(dfList=[mount1, mount2, mount3, mount4])
    sc_df = concat_data(dfList=[sc1, sc2, sc3])
    jits_df = concat_data(dfList=[general_jits1, general_jits2, general_jits3])
    
    mount_avg_df = build_structure(mount_df)
    sc_avg_df = build_structure(sc_df)
    jits_avg_df = build_structure(jits_df)
    
    mount_avg_df = set_state(mount_avg_df, 'mount')
    sc_avg_df = set_state(sc_avg_df, 'side control')
    jits_avg_df = set_state(jits_avg_df, 'jits')

    final_df = concat_data(dfList=[mount_avg_df, sc_avg_df, jits_avg_df])

    return final_df

def prep_mix_data():
    mix = fetch_data(54)
    mix_avg = build_structure(mix)

    print 'prep_mix_data: {}'.format(np.where(mix_avg['x_rolling_average'].isnull())[0]) 

    x1 = mix_avg['ACCELEROMETER_X'].values
    x2 = mix_avg['ACCELEROMETER_Y'].values
    x3 = mix_avg['ACCELEROMETER_Z'].values
    x4 = mix_avg['x_rolling_average'].values
    x5 = mix_avg['y_rolling_average'].values
    x6 = mix_avg['z_rolling_average'].values
    x7 = mix_avg['x_rolling_median'].values
    x8 = mix_avg['y_rolling_median'].values
    x9 = mix_avg['z_rolling_median'].values
    x10 = mix_avg['x_max'].values
    x11 = mix_avg['y_max'].values
    x12 = mix_avg['z_max'].values
    x13 = mix_avg['x_diff'].values
    x14 = mix_avg['y_diff'].values
    x15 = mix_avg['z_diff'].values


    X = np.column_stack([x1, x2, x3, x4, x5, x6, x10, x11, x12, x13, x14, x15])
    return X

def run_science():

    prepped_data = prep_data()

    # check for NaN values
    print 'Number of initial NAN values: {}'.format(np.where(prepped_data['x_rolling_average'].isnull())[0])
    
    prepped_data = prepped_data.drop(['id', 'index', 'timestamp'], axis=1)
    prepped_data = prepped_data.dropna()

    print 'Number of post-drop NAN values: {}'.format(np.where(prepped_data['x_rolling_average'].isnull())[0])

    print prepped_data
    model = run_model(prepped_data)


def rolling_average(df):
    return pd.rolling_mean(df, window=10, center=True).mean()

def rolling_median(df):
    return pd.rolling_median(df, window=10, center=True).mean()

def rolling_max(df):
    return pd.rolling_max(df, window=10).mean()

def start_diff(df):
    start = df[0:1]
    result = start - df
    #print 'start value: {}'.format(start)
    #print 'result value: {}'.format(result)
    return result

    """TODO

        COVARIANCE
        s1 = pd.Series(np.random.randn(1000))
        In [6]: s2 = pd.Series(np.random.randn(1000))
        In [7]: s1.cov(s2)

    """


def build_structure(df):
    # To begin, we create time sequences 
    original_x = df['ACCELEROMETER_X'].astype(float)
    original_y = df['ACCELEROMETER_Y'].astype(float)
    original_z = df['ACCELEROMETER_Z'].astype(float)
    i = original_x.index.values

    # initialize
    df['x_rolling_average'] = np.nan
    df['y_rolling_average'] = np.nan
    df['z_rolling_average'] = np.nan
    df['x_rolling_median'] = np.nan
    df['y_rolling_median'] = np.nan
    df['z_rolling_median'] = np.nan
    df['x_max'] = np.nan
    df['y_max'] = np.nan
    df['z_max'] = np.nan
    df['x_diff'] = np.nan
    df['y_diff'] = np.nan
    df['z_diff'] = np.nan

    # TODO: seems very inelegant
    idx = np.array([i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]).T.flatten()[:len(original_x)]
    x = original_x.groupby(idx).mean()
    y = original_y.groupby(idx).mean()
    z = original_z.groupby(idx).mean()

    x_ra = original_x.groupby(idx).apply(rolling_average)
    y_ra = original_y.groupby(idx).apply(rolling_average)
    z_ra = original_z.groupby(idx).apply(rolling_average)

    x_rm = original_x.groupby(idx).apply(rolling_median)
    y_rm = original_y.groupby(idx).apply(rolling_median)
    z_rm = original_z.groupby(idx).apply(rolling_median)

    x_max = original_x.groupby(idx).apply(rolling_max)
    y_max = original_y.groupby(idx).apply(rolling_max)
    z_max = original_z.groupby(idx).apply(rolling_max)

    zscore = lambda x: (x - x.mean()) / x.std()
    x_diff = original_x.groupby(idx).transform(zscore)
    y_diff = original_y.groupby(idx).transform(zscore)
    z_diff = original_z.groupby(idx).transform(zscore)

    avg_df = pd.DataFrame(columns=df.columns)

    avg_df['ACCELEROMETER_X'] = x
    avg_df['ACCELEROMETER_Y'] = y
    avg_df['ACCELEROMETER_Z'] = z
    avg_df['x_rolling_average'] = x_ra
    avg_df['y_rolling_average'] = y_ra
    avg_df['z_rolling_average'] = z_ra
    avg_df['x_rolling_median'] = x_rm
    avg_df['y_rolling_median'] = y_rm
    avg_df['z_rolling_median'] = z_rm
    avg_df['x_max'] = x_max
    avg_df['y_max'] = y_max
    avg_df['z_max'] = z_max
    avg_df['x_diff'] = x_diff
    avg_df['y_diff'] = y_diff
    avg_df['z_diff'] = z_diff

    # TODO: fix so that it doesn't just record the first experiment_id
    avg_df['experiment_id'] = df.experiment_id

    return avg_df

        
def set_state(df, state):
    """set the state for training
    right now I am using:

    1 = mount
    2 = side control
    3 = general jits
    """

    if state == 'mount':
        df.state = 1
    elif state == 'side control':
        df.state = 2
    elif state =='jits':
        df.state = 3

    return df

    

def run_model(df):
    df_train = df

    x1 = df_train['ACCELEROMETER_X'].values
    x2 = df_train['ACCELEROMETER_Y'].values
    x3 = df_train['ACCELEROMETER_Z'].values

    x4 = df_train['x_rolling_average'].values
    x5 = df_train['y_rolling_average'].values
    x6 = df_train['z_rolling_average'].values 

    # 0.77 accuracy

    x7 = df_train['x_rolling_median'].values
    x8 = df_train['y_rolling_median'].values
    x9 = df_train['z_rolling_median'].values

    x10 = df_train['x_max'].values
    x11 = df_train['y_max'].values
    x12 = df_train['z_max'].values

    x13 = df_train['x_diff'].values
    x14 = df_train['y_diff'].values
    x15 = df_train['z_diff'].values

    # still at 0.74 accuracy

    print "here in run_model: {}".format(np.where(df_train['x_rolling_average'].isnull())[0])

    y = df_train['state'].values
    X = np.column_stack([x1, x2, x3, x4, x5, x6, x10, x11, x12, x13, x14, x15])

    clf = svm.SVC(kernel='linear', C=1)
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

    # figure out optimal n_estimators

    n_range = range(1, 50)
    start_range = [True, False]

    param_grid = dict(n_estimators=n_range, warm_start=start_range)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)

    grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
    print grid_mean_scores

    print grid.best_score_
    print grid.best_params_
    print grid.best_estimator_


    # We can pass in the entirety of the data to cross_val_score, it takes care of splitting data into test/train
    # http://blog.kaggle.com/2015/06/29/scikit-learn-video-7-optimizing-your-model-with-cross-validation/
    # http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html

    # REAL TEST

    
    test_data = prep_mix_data()

    print test_data.shape


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    clf.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)
    knn_pred = knn.predict(X_test)
    rf_pred = rf.predict(X_test)

    rf_pred2 = rf.predict(test_data)
    print_full(rf_pred2)

    print 'SVM prediction: {}'.format(accuracy_score(y_test, clf_pred)) 
    print 'knn prediction: {}'.format(accuracy_score(y_test, knn_pred)) 
    print 'rf prediction: {}'.format(accuracy_score(y_test, rf_pred))

    svm_scores = cross_validation.cross_val_score(
        clf, X, df_train.state, cv=10, scoring='accuracy')

    print("SVM Accuracy: %0.2f (+/- %0.2f)" % (svm_scores.mean(), svm_scores.std() * 2))

    knn_scores = cross_validation.cross_val_score(
        knn, X, df_train.state, cv=10, scoring='accuracy')

    print("KNN Accuracy: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))

    rf_scores = cross_validation.cross_val_score(
        rf, X, df_train.state, cv=10, scoring='accuracy')


    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))
    
    #return clf


    





