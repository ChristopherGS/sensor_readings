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

from flask import (Blueprint, current_app, Markup, Response, abort, flash, jsonify,
                   make_response, redirect, render_template, request, send_from_directory, session, url_for)

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
    #import pdb; pdb.set_trace();
    
    if android_status == True:
        current_app.logger.debug('processing Android file')
        current_app.logger.debug(file_name)
        df = pd.read_csv(file_name, index_col=False, skipinitialspace=True, encoding='utf-8')

        data = df
        data['X_AXIS'] = data['X_AXIS'].astype(str)
        data['Y_AXIS'] = data['Y_AXIS'].astype(str)
        data['Z_AXIS'] = data['Z_AXIS'].astype(str)
        data['TIMESTAMP'] = data['TIMESTAMP'].astype(str)
        data['TIMESTAMP'] = data['TIMESTAMP'].apply(lambda s: s.replace('-', ''))
 
        data['X_AXIS'] = data['X_AXIS'].apply(lambda s: s.replace('(', ''))
        data['X_AXIS'] = data['X_AXIS'].apply(lambda s: s.replace(')', ''))
        data['Y_AXIS'] = data['Y_AXIS'].apply(lambda s: s.replace('(', ''))
        data['Y_AXIS'] = data['Y_AXIS'].apply(lambda s: s.replace(')', ''))
        data['Z_AXIS'] = data['Z_AXIS'].apply(lambda s: s.replace('(', ''))
        data['Z_AXIS'] = data['Z_AXIS'].apply(lambda s: s.replace(')', ''))
        
        #data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='ms', errors='coerce')
        data['TIMESTAMP'] = data['TIMESTAMP'].apply(lambda x:datetime.strptime(x,"%Y%m%d%H%M%S%f"))

        current_app.logger.debug(data)
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


# TEMP: In production, you don't want to server static files using the flask server.
"""
class DownloadCsv(Resource):
    def get(self, id):
        return{"message":id}, 200
    def post(self, id):
        #import pdb; pdb.set_trace()
        print 'generating csv for experiment: {}'.format(id)
        try: 
            print id
            query = db.session.query(Sensor)
            df = pd.read_sql_query(query.statement, query.session.bind)
            pandas_id = id
            df2 = df[df.experiment_id == pandas_id]
            filename = 'mbient_{}.csv'.format(id)
            columns = df2.columns
            df2.to_csv(os.path.join(UPLOAD_FOLDER, filename))
            #csv = df2.to_csv(filename, columns=columns)
            return send_from_directory(directory=UPLOAD_FOLDER, filename=filename, as_attachment=True)
        except Exception as e:
            current_app.logger.debug('error: {}'.format(e))
            abort(500)
"""

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
                    ANDROID = True
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
        