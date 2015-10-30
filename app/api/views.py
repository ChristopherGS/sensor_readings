import os
import pandas as pd
import numpy as np
from app.sensors.models import Experiment, Sensor
from werkzeug import secure_filename
from datetime import datetime
import csv
import json
import os
from datetime import datetime
from time import mktime

from flask import (Blueprint, current_app, Markup, Response, abort, flash, jsonify,
                   redirect, render_template, request, session, url_for)

from flask_restful import reqparse, Resource

from numpy import genfromtxt
from werkzeug import secure_filename
from werkzeug.exceptions import default_exceptions, HTTPException

from app.data import db, query_to_list
from app.science import pandas_cleanup, sql_to_pandas

_basedir = os.path.abspath(os.path.dirname(__file__))

ALLOWED_EXTENSIONS = set(['txt', 'csv'])
UPLOADS = 'uploads'
UPLOAD_FOLDER = os.path.join(_basedir, UPLOADS)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def Load_Data(file_name, android_status):
    #import pdb; pdb.set_trace()
    if android_status == True:
        current_app.logger.debug('processing Android file')
        df = pd.read_csv(file_name, names=['initial'])

        data = pd.DataFrame(df.index.tolist(), columns=['ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z'])
        data = data.applymap(str)
        data = data.apply(lambda s: s.str.replace('(', ''))
        data = data.apply(lambda s: s.str.replace(')', ''))

        current_app.logger.debug(data)
        #sLength = len(data['initial'])
        # split the tuples

        print data.index

        #data.loc[:,'ACCELEROMETER_X'] = pd.Series(np.random.randn(sLength), index=data.index)
        #data.loc[:,'ACCELEROMETER_Y'] = pd.Series(np.random.randn(sLength), index=data.index)
        #data.loc[:,'ACCELEROMETER_Z'] = pd.Series(np.random.randn(sLength), index=data.index)

        data = data.reindex(columns=['ACCELEROMETER_X',
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
            'timestamp'])

    else:
        data = pd.read_table(file_name, header=None, skiprows=1, delimiter=',', encoding='iso-8859-1') 

    return data.values.tolist()


def process_time(unknown_timestamp):
    try:
        final_timestamp = datetime.strptime(unknown_timestamp, "%Y-%m-%d %H:%M:%S:%f")
    except Exception as e:
        current_app.logger.debug('Error changing timestamp {}'.format(e))
        # TODO: THIS IS A RISKY HACK!
        final_timestamp = datetime.now()
    return final_timestamp

# API views

class CsvSimple(Resource):

    def get(self):
        return {"message":"hello"}, 200
    def post(self):
        # first test if the data is anything at all
        try:
            all_data = request.get_data()
            current_app.logger.debug('____api received_____ {}'.format(all_data))
            
        except Exception as e:
            current_app.logger.debug('error: {}'.format(e))
            return {"error": e}, 500

        # now test if it is a file or not
        ANDROID = False
        try:
            if request.files['android_file']:
                file = request.files['android_file']
                ANDROID = True
            elif request.files['a_file']:
                file = request.files['a_file']
                
            else:
                current_app.logger.debug('file recognition failed')
                raise AssertionError("Unexpected value of request files", request.files)

        except Exception as e:
            current_app.logger.debug('this is not a file: {}'.format(e))
            current_app.logger.debug(request.form)
            return {"error":"hey can you send a file"}, 500

        if file and allowed_file(file.filename):
            print "allowed_file"
            
            filename = secure_filename(file.filename)
            file.save(UPLOAD_FOLDER + '/' + filename)

            my_experiment = Experiment(hardware='Nexus5', t_stamp=datetime.now(), label='unknown')
            db.session.add(my_experiment)
            db.session.commit()

            try:
                file_name = filename
                
                # note that Load_Data expects a correctly formatted file - expects columns
                data = Load_Data(os.path.join(UPLOAD_FOLDER, file_name), ANDROID)
                current_app.logger.debug(data)
                count = 0 

                for i in data:
                    my_timestamp = process_time(i[19])
                    el_sensor = Sensor(**{
                        'ACCELEROMETER_X' : i[0],
                        'ACCELEROMETER_Y' : i[1],
                        'ACCELEROMETER_Z' : i[2],
                        'timestamp' : my_timestamp,
                        'experiment' : my_experiment
                    })
                    
                    db.session.add(el_sensor)
                    count += 1
                
                current_app.logger.debug('Committing {} records to the database'.format(count))
                db.session.commit() 
                flash("CSV data saved to DB") 
                return {'message': 'data saved to db'}, 201

            except Exception as e:
                print e
                db.session.rollback()
                # TODO SEND INFO BACK TO DEVICE
                return {'error': e}, 500

        else:
            current_app.logger.debug('Not a .txt or .csv file')
            return {'message':'incorrect file format'}, 500