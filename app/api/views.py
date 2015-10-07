import os
import pandas as pd
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

def Load_Data(file_name):
    # data = genfromtxt(file_name, delimiter=',', skiprows=1)
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
        try:
            all_data = request.get_data()
            current_app.logger.debug('____api received_____ {}'.format(all_data))
            file = request.files['file']
        except Exception as e:
            return {"error": e}, 500

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
                data = Load_Data(os.path.join(UPLOAD_FOLDER, file_name))
                count = 0 

                # need to handle this for when the uploaded file has less than 3 columns

                for i in data:
                    u = Experiment.query.get(1)
                    # my_timestamp = datetime.strptime(i[3], "%Y-%m-%d %H:%M:%S:%f")
                    my_timestamp = process_time(i[3])
                    el_sensor = Sensor(**{
                        'accelerometer_x' : i[0],
                        'accelerometer_y' : i[1],
                        'accelerometer_z' : i[2],
                        'timestamp' : my_timestamp,
                        'experiment' : my_experiment
                    })
                    
                    db.session.add(el_sensor)
                    count += 1
                
                # import pdb; pdb.set_trace()
                current_app.logger.debug('Committing {} records to the database'.format(count))
                db.session.commit() #Attempt to commit all the records
                flash("CSV data saved to DB") 
                return {'message': 'data saved to db'}, 201

            except Exception as e:
                print e
                db.session.rollback() #Rollback the changes on error
                # TODO SEND INFO BACK TO DEVICE
                return {'error': e}, 500