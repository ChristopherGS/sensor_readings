from __future__ import division
import numpy as np
import pandas as pd
import os
import glob
from dateutil import parser
from datetime import datetime

import config


DIR = os.path.dirname(os.path.realpath(__file__))


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

def concat_data(dfList = [], *args):
    data = [x for x in dfList]
    complete_df = pd.concat(data, axis=0)
    complete_df = complete_df.reset_index()
    return complete_df


def resolve_acc_gyro(df):
    """
    combine separate accelerometer and gyrocope rows into one row
    note that this has changed from older data
    """

    df.drop(df.columns[[0, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22]], axis=1, inplace=True)
    # TODO: Note that this is adding 8 hours to the time - mistakently thinks timezone is off. 
    df['timestamp'] = df['timestamp'].apply(lambda x: parser.parse(x))
    # TO DISCUSS
    # we will try merging the data based on timestamp, but we need to be forgiving to allow for small differences, 
    # so we round the microseconds
    df['timestamp'] = df['timestamp'].apply(lambda x: format_time(x))
    df_accel = df[df['SENSOR_TYPE'] == 'Accelerometer'].copy()

    df_accel['ACCEL_X'] = df_accel['X_AXIS']
    df_accel['ACCEL_Y'] = df_accel['Y_AXIS']
    df_accel['ACCEL_Z'] = df_accel['Z_AXIS']

    df_gyro = df[df['SENSOR_TYPE'] == 'Gyro'].copy()

    df_gyro['GYRO_X'] = df_gyro['X_AXIS']
    df_gyro['GYRO_Y'] = df_gyro['Y_AXIS']
    df_gyro['GYRO_Z'] = df_gyro['Z_AXIS']

    df2 = pd.merge(df_accel, df_gyro, how='outer', on=['Time_since_start']) # previously used timestamp

    # having done the merge, mark as NaN any rows which do not have *both* accelerometer and gyro data

    df2.replace(r'\s+', np.nan, regex=True)
    df2.drop(df2.columns[[0, 1, 2, 3, 4, 9, 10, 11, 12]], axis=1, inplace=True)
    return df2


def combine_csv(directory_description):
    """concatenate multiple csv files into one pandas dataframe"""

    

    allFiles = glob.glob(DIR + '/data/'+ directory_description + '/*.csv')
    print allFiles
    df = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        df = resolve_acc_gyro(df)
        list_.append(df)
    df = pd.concat(list_)
    complete_df = df.reset_index()
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

def convert_to_words(df):

    updated_df = ['your_mount' if v == 0 
                    else 'your_side_control' if v == 1
                    else 'your_closed_guard' if v == 2
                    else 'your_back_control' if v == 3
                    else 'opponent_mount_or_sc' if v == 4
                    else 'opponent_closed_guard' if v == 5
                    else 'opponent_back_control' if v == 6
                    else 'OTHER' if v == 7
                    else 'UNKNOWN' for v in df]

    return updated_df

def get_position_stats(df):
    total = len(df)

    try: 
        ymount = (df.count('your_mount'))/total
    except ZeroDivisionError:
        ymount = 0

    try: 
        ysc = (df.count('your_side_control'))/total
    except ZeroDivisionError:
        ysc = 0

    try: 
        ycg = (df.count('your_closed_guard'))/total
    except ZeroDivisionError:
        ycg = 0

    try: 
        ybc = (df.count('your_back_control'))/total
    except ZeroDivisionError:
        ybc = 0

    try: 
        omountsc = (df.count('opponent_mount_or_sc'))/total
    except ZeroDivisionError:
        omountsc = 0

    try: 
        ocg = (df.count('opponent_closed_guard'))/total
    except ZeroDivisionError:
        ocg = 0

    try: 
        obc = (df.count('opponent_back_control'))/total
    except ZeroDivisionError:
        obc = 0

    try: 
        OTHER = (df.count('OTHER'))/total
    except ZeroDivisionError:
        OTHER = 0


    print ('Your Mount: {0}\n'
            'Your Side Control: {1}\n'
            'Your Closed Guard: {2}\n'
            'Your Back Control: {3}\n'
            'Opponent Mount or Opponent Side Control: {4}\n'
            'Opponent Closed Guard: {5}\n'
            'Opponent Back Control: {6}\n'
            'OTHER: {7}\n'
            .format(ymount, ysc, ycg, ybc,
                omountsc, ocg, obc, OTHER))