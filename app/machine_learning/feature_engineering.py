import numpy as np
import pandas as pd
import os

TIME_SEQUENCE_LENGTH = 30


def rolling_average(df):
    return pd.rolling_mean(df, window=TIME_SEQUENCE_LENGTH-2, center=True).mean()

def rolling_median(df):
    return pd.rolling_median(df, window=TIME_SEQUENCE_LENGTH-2, center=True).mean()

def rolling_max(df):
    return pd.rolling_max(df, window=TIME_SEQUENCE_LENGTH-2, center=True).mean()

def rolling_min(df):
    return pd.rolling_min(df, window=TIME_SEQUENCE_LENGTH-2, center=True).mean()

def standard_deviation(df):
    std = pd.rolling_std(df, window=TIME_SEQUENCE_LENGTH-2, center=True)
    print std
    return std

def max_min_dif(df):
    diff = (pd.rolling_min(df, window=TIME_SEQUENCE_LENGTH-2, center=True).mean()) - (pd.rolling_max(df, window=TIME_SEQUENCE_LENGTH-2, center=True).mean())
    return diff

def create_rm_feature(df, sequence_length):
    features = []

    original_x = df['ACCEL_X'].astype(float)
    original_y = df['ACCEL_Y'].astype(float)
    original_z = df['ACCEL_Z'].astype(float)
    original_gx = df['GYRO_X'].astype(float)
    original_gy = df['GYRO_Y'].astype(float)
    original_gz = df['GYRO_Z'].astype(float)

    i = original_x.index.values
    time_sequence = [i] * sequence_length

    # The acutal number of 'i' values in the np.array is the sequence_length

    # COULD try putting a Kalman filter here before we split into 10 second chunks
    # this would smooth all the readings before processing - not sure if that is a good approach...
    # This filter could replace values or form new features

    idx = np.array(time_sequence).T.flatten()[:len(original_x)]
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
    
    #rolling median

    x_ra = df['ACCEL_X'].groupby(idx).apply(rolling_median)
    x_ra.name = 'rolling_median_x'
    features.append(x_ra)

    y_ra = df['ACCEL_Y'].groupby(idx).apply(rolling_median)
    y_ra.name = 'rolling_median_y'
    features.append(y_ra)

    z_ra = df['ACCEL_Z'].groupby(idx).apply(rolling_median)
    z_ra.name = 'rolling_median_z'
    features.append(z_ra)

    #gx_ra = df['GYRO_X'].groupby(idx).apply(rolling_median)
    #gx_ra.name = 'rolling_median_gx'
    #features.append(gx_ra)

    #gy_ra = df['GYRO_Y'].groupby(idx).apply(rolling_median)
    #gy_ra.name = 'rolling_median_gy'
    #features.append(gy_ra)

    #gz_ra = df['GYRO_Z'].groupby(idx).apply(rolling_median)
    #gz_ra.name = 'rolling_median_gz'
    #features.append(gz_ra)

    #rolling max

    x_rm = df['ACCEL_X'].groupby(idx).apply(rolling_max)
    x_rm.name = 'rolling_max_x'
    features.append(x_rm)

    y_rm = df['ACCEL_Y'].groupby(idx).apply(rolling_max)
    y_rm.name = 'rolling_max_y'
    features.append(y_rm)

    z_rm = df['ACCEL_Z'].groupby(idx).apply(rolling_max)
    z_rm.name = 'rolling_max_z'
    features.append(z_rm)

    #gx_rm = df['GYRO_X'].groupby(idx).apply(rolling_max)
    #gx_rm.name = 'rolling_max_gx'
    #features.append(gx_rm)

    #gy_rm = df['GYRO_Y'].groupby(idx).apply(rolling_max)
    #gy_rm.name = 'rolling_max_gy'
    #features.append(gy_rm)

    #gz_rm = df['GYRO_Z'].groupby(idx).apply(rolling_max)
    #gz_rm.name = 'rolling_max_gz'
    #features.append(gz_rm)

    #rolling min

    x_rmin = df['ACCEL_X'].groupby(idx).apply(rolling_min)
    x_rmin.name = 'rolling_min_x'
    features.append(x_rmin)

    y_rmin = df['ACCEL_Y'].groupby(idx).apply(rolling_min)
    y_rmin.name = 'rolling_min_y'
    features.append(y_rmin)

    z_rmin = df['ACCEL_Z'].groupby(idx).apply(rolling_min)
    z_rmin.name = 'rolling_min_z'
    features.append(z_rmin)

    #gx_rmin = df['GYRO_X'].groupby(idx).apply(rolling_min)
    #gx_rmin.name = 'rolling_min_gx'
    #features.append(gx_rmin)

    #gy_rmin = df['GYRO_Y'].groupby(idx).apply(rolling_min)
    #gy_rmin.name = 'rolling_min_gy'
    #features.append(gy_rmin)

    #gz_rmin = df['GYRO_Z'].groupby(idx).apply(rolling_min)
    #gz_rmin.name = 'rolling_min_gz'
    #features.append(gz_rmin)

    #standard deviation

    #x_std = df['ACCEL_X'].groupby(idx).apply(standard_deviation)
    #x_std.name = 'std_x'
    #features.append(x_std)

    #y_std = df['ACCEL_Y'].groupby(idx).apply(standard_deviation)
    #y_std.name = 'std_y'
    #features.append(y_std)

    #z_std = df['ACCEL_Z'].groupby(idx).apply(standard_deviation)
    #z_std.name = 'std_z'
    #features.append(z_std)

    #gx_std = df['GYRO_X'].groupby(idx).apply(standard_deviation)
    #gx_std.name = 'std_gx'
    #features.append(gx_std)

    #gy_std = df['GYRO_Y'].groupby(idx).apply(standard_deviation)
    #gy_std.name = 'std_gy'
    #features.append(gy_std)

    #gz_std = df['GYRO_Z'].groupby(idx).apply(standard_deviation)
    #gz_std.name = 'std_gz'
    #features.append(gz_std)

    # Max min diff

    x_diff = df['ACCEL_X'].groupby(idx).apply(max_min_dif)
    x_diff.name = 'diff_x'
    features.append(x_diff)

    y_diff = df['ACCEL_Y'].groupby(idx).apply(max_min_dif)
    y_diff.name = 'diff_y'
    features.append(y_diff)

    z_diff = df['ACCEL_Z'].groupby(idx).apply(max_min_dif)
    z_diff.name = 'diff_z'
    features.append(z_diff)

    gx_diff = df['GYRO_X'].groupby(idx).apply(max_min_dif)
    gx_diff.name = 'diff_gx'
    features.append(gx_diff)

    gy_diff = df['GYRO_Y'].groupby(idx).apply(max_min_dif)
    gy_diff.name = 'diff_gy'
    features.append(gy_diff)

    gz_diff = df['GYRO_Z'].groupby(idx).apply(max_min_dif)
    gz_diff.name = 'diff_gz'
    features.append(gz_diff)

    data = pd.concat(features, axis=1)
    return data