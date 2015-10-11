import csv
import os
import sys
import requests

first_arg = sys.argv[1]
second_arg = sys.argv[2]
URL = 'http://christophergs.pythonanywhere.com/api/csv'

inputFileName = second_arg
outputFileName = os.path.splitext(inputFileName)[0] + "_modified.csv"

def prep(inputFileName):

    try:

        with open(inputFileName, 'rb') as inFile, open(outputFileName, 'wb') as outfile:
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

            print 'It worked, modified file created.'

            # Now send the file to the server

    except Exception as e:
        print 'Error, something went wrong: {}'.format(e)

    
    try: 
        filehandle = open(outputFileName)
        r = requests.post(URL, files={'a_file':filehandle})
        print r.text
    except Exception as e:
        print 'error, file not uploaded: {}'.format(e)
    

if __name__ == "__main__":
    if first_arg == 'prep':
        prep(second_arg)
    else:
        print 'no such function, fool.'