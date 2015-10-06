

from flask import Flask, render_template, request
from werkzeug.exceptions import default_exceptions, HTTPException

from flask_wtf.csrf import CsrfProtect
from flask_restful import Resource, Api


from .data import db

from .sensors.views import sensors
from .sensors import views
from .users.views import users

import app.errors
# import app.logs

# temp

import os
import pandas as pd
from .sensors.models import Experiment, Sensor
from werkzeug import secure_filename
from datetime import datetime
import csv
import json
import os
from datetime import datetime
from time import mktime

from flask import (Blueprint, current_app, Markup, Response, abort, flash, jsonify,
                   redirect, render_template, request, session, url_for)
from numpy import genfromtxt
from werkzeug import secure_filename
from werkzeug.exceptions import default_exceptions, HTTPException

from .data import db, query_to_list
from .science import pandas_cleanup, sql_to_pandas

app = Flask(__name__)
app.config.from_object('config.DebugConfiguration')


@app.context_processor
def provide_constants():
    return {"constants": {"TUTORIAL_PART": 2}}

db.init_app(app)
csrf_protect = CsrfProtect(app)
api = Api(app, decorators=[csrf_protect.exempt])

app.register_blueprint(sensors)
app.register_blueprint(users)

_basedir = os.path.abspath(os.path.dirname(__file__))


ALLOWED_EXTENSIONS = set(['txt', 'csv'])
UPLOADS = 'uploads'
UPLOAD_FOLDER = os.path.join(_basedir, UPLOADS)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def Load_Data(file_name):
    # data = genfromtxt(file_name, delimiter=',', skiprows=1)
    data = pd.read_table(file_name, header=None, skiprows=1, delimiter=',') 
    return data.values.tolist()


def process_time(unknown_timestamp):
    try:
        final_timestamp = datetime.strptime(unknown_timestamp, "%Y-%m-%d %H:%M:%S:%f")
    except Exception as e:
        app.logger.debug('Error changing timestamp {}'.format(e))
        # TODO: THIS IS A RISKY HACK!
        final_timestamp = datetime.now()
    return final_timestamp






@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404

@app.errorhandler(400)
def key_error(e):
    app.logger.warning('Invalid request resulted in KeyError', exc_info=e)
    return render_template('errors/400.html'), 400


@app.errorhandler(500)
def internal_server_error(e):
    app.logger.warning('An unhandled exception is being displayed to the end user', exc_info=e)
    return render_template('errors/generic.html'), 500


@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error('An unhandled exception is being displayed to the end user', exc_info=e)
    return render_template('errors/generic.html'), 500


errors.init_app(app)
# logs.init_app(app)

#----------------------------------------
# logging
#----------------------------------------

import logging


class ContextualFilter(logging.Filter):
    def filter(self, log_record):
        log_record.url = request.path
        log_record.method = request.method
        log_record.ip = request.environ.get("REMOTE_ADDR")

        return True

context_provider = ContextualFilter()
app.logger.addFilter(context_provider)
del app.logger.handlers[:]

handler = logging.StreamHandler()

log_format = "%(asctime)s\t%(levelname)s\t%(ip)s\t%(method)s\t%(url)s\t%(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)

app.logger.addHandler(handler)

from logging import ERROR
from logging.handlers import TimedRotatingFileHandler

# Only set up a file handler if we know where to put the logs
if app.config.get("ERROR_LOG_PATH"):

    # Create one file for each day. Delete logs over 7 days old.
    file_handler = TimedRotatingFileHandler(filename=app.config["ERROR_LOG_PATH"], when="D", backupCount=7)

    # Use a multi-line format for this logger, for easier scanning
    file_formatter = logging.Formatter('''
    Time: %(asctime)s
    Level: %(levelname)s
    Method: %(method)s
    Path: %(url)s
    IP: %(ip)s
    Message: %(message)s
    ---------------------''')

    # Filter out all log messages that are lower than Error.
    file_handler.setLevel(logging.DEBUG)

    # file_handler.addFormatter(file_formatter)
    app.logger.addHandler(file_handler)


# API views

class CsvSimple(Resource):
    def get(self):
        return {"message":"hello"}, 200
    def post(self):
        file = request.files['file']
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
                app.logger.debug('Committing {} records to the database'.format(count))
                db.session.commit() #Attempt to commit all the records
                flash("CSV data saved to DB") 
                return {'message': 'data saved to db'}, 201

            except Exception as e:
                print e
                db.session.rollback() #Rollback the changes on error
                # TODO SEND INFO BACK TO DEVICE
                return {'error': e}, 500
            



api.add_resource(CsvSimple, '/api/csv')