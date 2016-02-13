import numpy as np
import pandas as pd
import os


def rolling_average(df):
    return pd.rolling_mean(df, window=10, center=True).mean()

def rolling_median(df):
    return pd.rolling_median(df, window=2, center=True).mean()

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

    #TODO: curretly an ugly hack - need to connect to the sequence_length
    # for now, the acutal number of 'i' values in the np.array is the sequence_length

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