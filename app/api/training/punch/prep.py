import csv
import os
import sys

first_arg = sys.argv[1]
second_arg = sys.argv[2]

inputFileName = second_arg
outputFileName = os.path.splitext(inputFileName)[0] + "_modified.csv"

f = 'simple.csv'

def prep(filename):

	with open(filename, 'rb') as inFile, open(outputFileName, 'wb') as outfile:
		r = csv.reader(inFile, delimiter=',', quotechar='|')
		w = csv.writer(outfile)
		r.next()
		w.writerow(['ACCELEROMETER_X',
            'ACCELEROMETER_Y',
            'ACCELEROMETER_Z',
            'GRAVITY_X',
            'GRAVITY_Y',
            'GRAVITY_Z',
            'LINEAR_ACCELERATION_X',
            'LINEAR_ACCELERATION_Y',
            'LINEAR_ACCELERATION_Z',
            'GYROSCOPE_X',
            'GYROSCOPE_Y',
            'GYROSCOPE_Z', 
            'MAGNETIC_FIELD_X',
            'MAGNETIC_FIELD_Y',
            'MAGNETIC_FIELD_Z',
            'ORIENTATION_Z',
            'ORIENTATION_X',
            'ORIENTATION_Y',
            'Time_since_start',
            'timestamp'
            ])

		for row in r:
			w.writerow(row)


if __name__ == "__main__":
    if first_arg == 'prep':
    	prep(second_arg)
    else:
    	print 'no such file, fool.'