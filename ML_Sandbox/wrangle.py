import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime
from dateutil import parser

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
    df_accel = df.loc[df['SENSOR_TYPE'] == 'Accel_Log']

    df_accel['ACCEL_X'] = pd.Series(df_accel['X_AXIS'], index=df_accel.index)
    df_accel['ACCEL_Y'] = pd.Series(df_accel['Y_AXIS'], index=df_accel.index)
    df_accel['ACCEL_Z'] = pd.Series(df_accel['Z_AXIS'], index=df_accel.index)

    df_gyro = df.loc[df['SENSOR_TYPE'] == 'Gyro_Log']

    df_gyro['GYRO_X'] = pd.Series(df_gyro['X_AXIS'], index=df_gyro.index)
    df_gyro['GYRO_Y'] = pd.Series(df_gyro['Y_AXIS'], index=df_gyro.index)
    df_gyro['GYRO_Z'] = pd.Series(df_gyro['Z_AXIS'], index=df_gyro.index)

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
    pass

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

def prep():
    """prepare the raw sensor data"""

    mount_training_data = combine_csv('mount_raw_data')
    mtd = set_state(mount_training_data, 'mount')

    # repeat to give the option to train on non-sequenced data
    mtd_mod = create_time_sequences(mtd)
    mtd_mod = set_state(mtd_mod, 'mount')

    side_control_training_data = combine_csv('side_control_raw_data')
    sctd = set_state(side_control_training_data, 'side control')
    sctd_mod = create_time_sequences(sctd)
    sctd_mod = set_state(sctd_mod, 'side control')
    
    general_jj_training_data = combine_csv('general_jj_raw_data')
    jjtd = set_state(general_jj_training_data, 'jits')
    jjtd_mod = create_time_sequences(jjtd)
    jjtd_mod = set_state(jjtd_mod, 'jits')

    training_data = concat_data([mtd_mod, sctd_mod, jjtd_mod])

training_data = prep()
print training_data





