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
import dateutil.parser
from time import mktime
import six.moves.cPickle as pickle

from flask import (Blueprint, current_app, Markup, Response, abort, flash, jsonify,
                   make_response, redirect, render_template, request, send_from_directory, session, url_for)

from flask_restful import reqparse, Resource

from numpy import genfromtxt
from werkzeug import secure_filename
from werkzeug.exceptions import default_exceptions, HTTPException
from sklearn.externals import joblib

from app.data import db, query_to_list
from app.science import pandas_cleanup, sql_to_pandas
from app.machine_learning.wrangle import api_serialize, api_test
from app.machine_learning.utilities import convert_to_words, get_position_stats

_basedir = os.path.abspath(os.path.dirname(__file__))

ALLOWED_EXTENSIONS = set(['txt', 'csv'])
UPLOADS = 'uploads'
UPLOAD_FOLDER = os.path.join(_basedir, UPLOADS)
PICKLE = os.path.join(_basedir, '../machine_learning/pickle/training.pkl')

if (os.path.exists(PICKLE)):
    algorithm = joblib.load(PICKLE)
else:
    print "Model fitting in progress"

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def Load_Data(file_name, android_status):
    
    if android_status == True:
        current_app.logger.debug('processing Android file')
        current_app.logger.debug(file_name)
        df = pd.read_csv(file_name, index_col=False, skipinitialspace=True, encoding='utf-8')

        data = df
        data['x_acceleration'] = data['x_acceleration'].astype(str)
        data['y_acceleration'] = data['y_acceleration'].astype(str)
        data['z_acceleration'] = data['z_acceleration'].astype(str)
        data['TIMESTAMP'] = data['TIMESTAMP'].astype(str)
        data['TIMESTAMP'] = data['TIMESTAMP'].apply(lambda s: s.replace('-', ''))
 
        data['x_acceleration'] = data['x_acceleration'].apply(lambda s: s.replace('(', ''))
        data['x_acceleration'] = data['x_acceleration'].apply(lambda s: s.replace(')', ''))
        data['y_acceleration'] = data['y_acceleration'].apply(lambda s: s.replace('(', ''))
        data['y_acceleration'] = data['y_acceleration'].apply(lambda s: s.replace(')', ''))
        data['z_acceleration'] = data['z_acceleration'].apply(lambda s: s.replace('(', ''))
        data['z_acceleration'] = data['z_acceleration'].apply(lambda s: s.replace(')', ''))
        
        #data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='ms', errors='coerce')
        data['TIMESTAMP'] = data['TIMESTAMP'].apply(lambda x:datetime.strptime(x,"%Y%m%d%H%M%S%f"))

        current_app.logger.debug(data)
    else:
        #data = pd.read_table(file_name, header=None, skiprows=1, delimiter=',', encoding='iso-8859-1') 
        current_app.logger.debug('processing non-Android file')
        current_app.logger.debug(file_name)
        df = pd.read_csv(file_name, index_col=False, skipinitialspace=True, encoding='utf-8')

        data = df
        data['X_AXIS'] = data['X_AXIS'].astype(str)
        data['Y_AXIS'] = data['Y_AXIS'].astype(str)
        data['Z_AXIS'] = data['Z_AXIS'].astype(str)
        data['timestamp'] = data['timestamp'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f"))

    return data.values.tolist()

def process_time(unknown_timestamp):
    try:
        final_timestamp = datetime.strptime(unknown_timestamp, "%Y-%m-%d %H:%M:%S:%f")
    except Exception as e:
        current_app.logger.debug('Error changing timestamp {}'.format(e))
        # TODO: THIS IS A RISKY HACK!
        final_timestamp = datetime.now()
    return final_timestamp


class CsvSimple(Resource):

    def get(self):
        return {"message":"hello"}, 200
    def post(self):
        # first test if the data is anything at all
        current_app.logger.debug('received post request')
        try:
            all_data = request.get_data()
            current_app.logger.debug('____api received_____ {}'.format(all_data))
            
        except Exception as e:
            current_app.logger.debug('error: {}'.format(e))
            return {"error": e}, 500

        # now test if it is a file or not
        ANDROID = False
        
        try:
            for my_file in request.files:

                if my_file == 'android_file':
                    file = request.files['android_file']
                    ANDROID = True
                elif my_file == 'a_file':
                    file = request.files['a_file']
                    ANDROID = False
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

                if ANDROID == True:
                    for i in data:
                        #my_timestamp = process_time(i[19])
                        el_sensor = Sensor(**{
                            'SENSOR_TYPE' : i[0],
                            'X_AXIS' : i[1],
                            'Y_AXIS' : i[2],
                            'Z_AXIS' : i[3],
                            'Time_since_start' : i[4],
                            'timestamp' : i[5],
                            'experiment' : my_experiment
                        })
                        
                        db.session.add(el_sensor)
                        count += 1
                else:
                    for i in data:
                        el_sensor = Sensor(**{
                            'SENSOR_TYPE' : i[2],
                            'X_AXIS' : i[6],
                            'Y_AXIS' : i[7],
                            'Z_AXIS' : i[8],
                            'Time_since_start' : i[18],
                            'timestamp' : i[20],
                            'experiment' : my_experiment
                        })
                        
                        db.session.add(el_sensor)
                        count += 1

                
                current_app.logger.debug('Committing {} records to the database'.format(count))
                db.session.commit() 
                flash("CSV data saved to DB")
                print 'HERE IS THE EXPERIMENT NUMBER'
                print my_experiment.id
                return {"id": my_experiment.id}, 201
                

            except Exception as e:
                print e
                db.session.rollback()
                # TODO SEND INFO BACK TO DEVICE
                return {'error': e}, 500

        else:
            current_app.logger.debug('Not a .txt or .csv file')
            return {'message':'incorrect file format'}, 500


class DataAnalysis(Resource):
    def get(self, experiment_id):
        try: 
            current_app.logger.debug('RECEIVED ANALYSIS REQUEST: {}'.format(experiment_id))
            print experiment_id
            test_data = api_test(experiment_id)
            current_app.logger.debug(test_data)
            predictions = algorithm.predict(test_data)
            current_app.logger.debug(predictions)
            converted_predictions = convert_to_words(predictions)
            stats = get_position_stats(converted_predictions)
            current_app.logger.debug(stats)
            print stats
            js = json.dumps(stats)
            resp = Response(js, status=200, mimetype='application/json')
            my_predictions = json.dumps(converted_predictions)
            current_app.logger.debug(my_predictions)
            return resp
        except Exception as e:
            current_app.logger.debug(e)
            return {'error': e}, 500
            
        