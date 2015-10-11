import csv
import json
import os
from datetime import datetime
from random import choice
from time import mktime

import pandas as pd
from flask import (Blueprint, current_app, Markup, Response, abort, flash, jsonify,
                   redirect, render_template, request, session, url_for)
from numpy import genfromtxt
from werkzeug import secure_filename
from werkzeug.exceptions import default_exceptions, HTTPException

from app.data import db, query_to_list
from app.science2 import pandas_cleanup, sql_to_pandas, my_svm

from .models import Experiment, Sensor

sensors = Blueprint("sensors", __name__, static_folder='static', template_folder='templates')

_basedir = os.path.abspath(os.path.dirname(__file__))

ALLOWED_EXTENSIONS = set(['txt', 'csv'])
UPLOADS = '../api/uploads'
UPLOAD_FOLDER = os.path.join(_basedir, UPLOADS)

@sensors.route("/")
@sensors.route("/index")
def index():
    return render_template("index.html")

@sensors.route('/csv')
def csv_route():
    return render_template('sensors/csv.html')
   
@sensors.route('/complete')
def complete():
     return render_template('sensors/complete.html')

@sensors.route('/guide')
def guide():
     return render_template('sensors/guide.html')

@sensors.route('/display', methods=['GET', 'POST'])
def display():
    if request.method == 'GET':
        sql_to_pandas() # TODO prep/check function

        # get csv files on server
        names = os.listdir(UPLOAD_FOLDER)

        # get sensor records from db
        query = db.session.query(Sensor)
        df = pd.read_sql_query(query.statement, query.session.bind)
        db_index = pd.unique(df.experiment_id.values)
        db_labels = []

        # loop through the given experiment_ids and pull out the labels from Experiment
        for i in db_index:
            experiment_labels = Experiment.query.filter_by(id=i).first()
            db_labels.append(experiment_labels.label)

        db_data = zip(db_index, db_labels)
        print db_data
            
        return render_template('sensors/show_files.html', file_url=names, db_data=db_data)
    elif request.method == 'POST':
        updated_label = request.values['label']
        db_id = request.values['id']

        # update experiment label in db

        experiment = Experiment.query.filter_by(id=db_id).first()
        experiment.label = updated_label
        current_app.logger.debug('Changing label in on a db object...')
        db.session.commit()

        return updated_label
    else:
        return '404'

@sensors.route('/display/<int:id>', methods=['GET', 'POST'])
def display_id(id):
    if request.method == 'GET':
        query = db.session.query(Sensor)
        df = pd.read_sql_query(query.statement, query.session.bind)
        pandas_id = id
        df2 = df[df.experiment_id == pandas_id]
        db_index_choice = df2[["ACCELEROMETER_X", "ACCELEROMETER_Y", "ACCELEROMETER_Z",
                                "GYROSCOPE_X", "GYROSCOPE_Y","GYROSCOPE_Z",
                                "Time_since_start", "state", "timestamp", "prediction"]]
        experiment_number = pd.unique(df2.experiment_id.values)
        return render_template('sensors/file_details.html', experiment_number=experiment_number[0], 
            db_index_choice=db_index_choice.to_html(), id=id)
    elif request.method == 'POST':
        try:
            updated_df = my_svm(id)         
            # save to DB
        except Exception as e:
            current_app.logger.debug('Error running model: {}'.format(e))
            return {'error': e}, 500

        return 'made a prediction', 200
    else:
        return '404'

@sensors.route('/display/<int:id>/graph')
def display_graph(id):

    query = db.session.query(Sensor)
    temp_df = pd.read_sql_query(query.statement, query.session.bind)
    
    # remove duplicate index values for json dump
    df = pandas_cleanup(temp_df) 
    
    pandas_id = id
    df2 = df[df.experiment_id == pandas_id]
    db_index_choice = df2
    experiment_number = pd.unique(df2.experiment_id.values)

    d3_json = df.to_json(orient='records')

    d3_response = json.dumps(d3_json)

    return render_template('sensors/file_graph.html', experiment_number=experiment_number[0], 
        db_index_choice=db_index_choice.to_html(), id=id, d3_response = d3_response)
