import math
import pandas as pd
import numpy as np

def root_sum_square(x, y, z):
        sum = ((x**2)+(y**2)+(z**2))
        rss = math.sqrt(sum)
        return rss

def root_mean_square(x, y, z):
        mean = ((x**2)+(y**2)+(z**2))/3
        rss = math.sqrt(mean)
        return rss

def tiltx(x, y, z):
    try:
        prep = (x/(math.sqrt((y**2)+(z**2))))
        tilt = math.atan(prep)
    except ZeroDivisionError:
        tilt = 0
    return tilt

def tilty(x, y, z):
    try:
        prep = (y/(math.sqrt((x**2)+(z**2))))
        tilt = math.atan(prep)
    except ZeroDivisionError:
        tilt = 0
    return tilt
    
def max_min_diff(max, min):
    diff = max - min
    return diff

def magnitude(x, y, z):
    magnitude = x + y + z
    return magnitude

def create_features(df, _window=50, test=False):
    """builds the data features, then applies
    overlapping logic
    """
    
    accel_x = df['ACCEL_X'].astype(float)
    accel_y = df['ACCEL_Y'].astype(float)
    accel_z = df['ACCEL_Z'].astype(float)
    gyro_x = df['GYRO_X'].astype(float)
    gyro_y = df['GYRO_Y'].astype(float)
    gyro_z = df['GYRO_Z'].astype(float)
    
    df2 = pd.DataFrame()
    
    # capture tilt here, then average later
    
    df2['tiltx'] = df.apply(lambda x: tiltx(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    df2['tilty'] = df.apply(lambda x: tilty(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    
    # Capture stand state here, then average later
    
    if (test==False):
        df2['stand'] = df['stand'].astype(float)
    
    TIME_SEQUENCE_LENGTH = _window
    
    # Basics
    
    df2['ACCEL_X'] = pd.rolling_mean(accel_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['ACCEL_Y'] = pd.rolling_mean(accel_y, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['ACCEL_Z'] = pd.rolling_mean(accel_z, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['GYRO_X'] = pd.rolling_mean(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['GYRO_Y'] = pd.rolling_mean(gyro_y, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['GYRO_Z'] = pd.rolling_mean(gyro_z, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # rolling median

    df2['rolling_median_x'] = pd.rolling_median(accel_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_median_y'] = pd.rolling_median(accel_y, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_median_z'] = pd.rolling_median(accel_z, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_median_gx'] = pd.rolling_median(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_median_gy'] = pd.rolling_median(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_median_gz'] = pd.rolling_median(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # rolling max
    
    df2['rolling_max_x'] = pd.rolling_max(accel_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_max_y'] = pd.rolling_max(accel_y, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_max_z'] = pd.rolling_max(accel_z, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_max_gx'] = pd.rolling_max(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_max_gy'] = pd.rolling_max(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_max_gz'] = pd.rolling_max(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # rolling min
    
    df2['rolling_min_x'] = pd.rolling_min(accel_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_min_y'] = pd.rolling_min(accel_y, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_min_z'] = pd.rolling_min(accel_z, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_min_gx'] = pd.rolling_min(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_min_gy'] = pd.rolling_min(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_min_gz'] = pd.rolling_min(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # rolling sum
    
    df2['rolling_sum_x'] = pd.rolling_sum(accel_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_sum_y'] = pd.rolling_sum(accel_y, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_sum_z'] = pd.rolling_sum(accel_z, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_sum_gx'] = pd.rolling_sum(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_sum_gy'] = pd.rolling_sum(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_sum_gz'] = pd.rolling_sum(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # standard deviation
    
    df2['rolling_std_x'] = pd.rolling_std(accel_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_std_y'] = pd.rolling_std(accel_y, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_std_z'] = pd.rolling_std(accel_z, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_std_gx'] = pd.rolling_std(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_std_gy'] = pd.rolling_std(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    df2['rolling_std_gz'] = pd.rolling_std(gyro_x, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # Tilt
    df2['avg_tiltx'] = pd.rolling_mean(df2['tiltx'], TIME_SEQUENCE_LENGTH-2, center=True)
    df2['avg_tilty'] = pd.rolling_mean(df2['tilty'], TIME_SEQUENCE_LENGTH-2, center=True)
    
    
    if (test==False):
        # standing up detection
        df2['avg_stand'] = pd.rolling_mean(df2['stand'], TIME_SEQUENCE_LENGTH-2, center=True)
        print df2['avg_stand']

        # round standing up as we need it to be either '0' or '1' for training later
        df2['avg_stand'] = df2['avg_stand'].apply(lambda x: math.ceil(x))

    ol_upper = _window/2
    ol_lower = ol_upper-1
        
    new_df = df2[ol_lower::ol_upper] # 50% overlap with 30
    
    new_df['max_min_x'] = df2.apply(lambda x: max_min_diff(x['rolling_max_x'], x['rolling_min_x']), axis=1)
    new_df['max_min_y'] = df2.apply(lambda x: max_min_diff(x['rolling_max_y'], x['rolling_min_y']), axis=1)
    new_df['max_min_z'] = df2.apply(lambda x: max_min_diff(x['rolling_max_z'], x['rolling_min_z']), axis=1)
    new_df['max_min_gx'] = df2.apply(lambda x: max_min_diff(x['rolling_max_gx'], x['rolling_min_gx']), axis=1)
    new_df['max_min_gy'] = df2.apply(lambda x: max_min_diff(x['rolling_max_gy'], x['rolling_min_gy']), axis=1)
    new_df['max_min_gz'] = df2.apply(lambda x: max_min_diff(x['rolling_max_gz'], x['rolling_min_gz']), axis=1)
                                                                       
    new_df['acc_rss'] = df2.apply(lambda x: root_sum_square(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    new_df['gyro_rss'] = df2.apply(lambda x: root_sum_square(x['GYRO_X'], x['GYRO_Y'], x['GYRO_Z']), axis=1)
    
    new_df['acc_rms'] = df2.apply(lambda x: root_mean_square(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    new_df['gyro_rms'] = df2.apply(lambda x: root_mean_square(x['GYRO_X'], x['GYRO_Y'], x['GYRO_Z']), axis=1)
    
    new_df['acc_magnitude'] = df2.apply(lambda x: magnitude(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    new_df['gyro_magnitude'] = df2.apply(lambda x: magnitude(x['GYRO_X'], x['GYRO_Y'], x['GYRO_Z']), axis=1)
        
    return new_df