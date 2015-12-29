import csv
import json
import os
from datetime import datetime
from random import choice
from time import mktime
from cStringIO import StringIO

import pandas as pd
from flask import (abort, Blueprint, current_app, Markup, Response, abort, flash, jsonify, make_response,
                   redirect, render_template, request, session, send_from_directory, url_for, stream_with_context)
from numpy import genfromtxt
from werkzeug import secure_filename
from werkzeug.exceptions import default_exceptions, HTTPException
from werkzeug.datastructures import Headers
#from werkzeug.wrappers import Response

from app.data import db, query_to_list
from app.science2 import pandas_cleanup, sql_to_pandas, my_svm, count_calculator
from app.science3 import run_science

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
     run_science()
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


#INITIAL!
@sensors.route('/display/blah')
def blah():
    query = db.session.query(Sensor)
    df = pd.read_sql_query(query.statement, query.session.bind)
    pandas_id = 36
    df2 = df[df.experiment_id == pandas_id]

    df2.to_csv(os.path.join(FOLDER, 'oi.csv'))
    filename = 'oi.csv'
    return send_from_directory(directory=FOLDER, filename=filename)


@sensors.route('/display/<int:id>', methods=['GET', 'POST', 'DELETE'])
def display_id(id):
    if request.method == 'GET':
        query = db.session.query(Sensor)
        df = pd.read_sql_query(query.statement, query.session.bind)
        pandas_id = id
        df2 = df[df.experiment_id == pandas_id]
        # add more columns here to display them
        db_index_choice = df2[["SENSOR_TYPE", "X_AXIS", "Y_AXIS",
                                "Z_AXIS",
                                "Time_since_start", "state", "timestamp", "prediction"]]
        experiment_number = pd.unique(df2.experiment_id.values)

        _info_label = Experiment.query.filter_by(id=id).first()
        info_label = _info_label.label

        # if prediction has been run, calculate summary stats
        print df2.prediction.values

        if (('straight punch' not in df2.prediction.values) & ('other' not in df2.prediction.values) & ('hook punch' not in df2.prediction.values)): 
            average = 'n/a'
            straight_average = 'n/a'
            hook_average = 'n/a'
            punch_count = 0
        else:
            calc_straight_punch = len(df2[df2['prediction']=='straight punch'])
            calc_hook_punch = len(df2[df2['prediction']=='hook punch'])
            calc_punches = calc_straight_punch + calc_hook_punch
            calc_total = len(df2['prediction'])
            punch_count = count_calculator(df2['prediction'])
            try:
                _average = (float(calc_punches)/float(calc_total))*100
                _straight_average = (float(calc_straight_punch)/float(calc_total))*100
                _hook_average = (float(calc_hook_punch)/float(calc_total))*100
                average = float("{0:.2f}".format(_average))
                straight_average = float("{0:.2f}".format(_straight_average))
                hook_average = float("{0:.2f}".format(_hook_average))
                current_app.logger.debug('Data has a {} chance of being a punch'.format(average))
            except Exception as e:
                if ((e == ZeroDivisionError) & (calc_punches == 0)):
                    average = 0
                    straight_average = 0
                    hook_average = 0
                else:
                    return render_template('errors/generic.html'), 500

        print average
        return render_template('sensors/file_details.html', experiment_number=experiment_number[0], 
            db_index_choice=db_index_choice.to_html(), id=id, average=average, straight_average=straight_average, 
            hook_average=hook_average, info_label=info_label, punch_count=punch_count)
    elif request.method == 'POST':
        try:
            updated_df = my_svm(id)         
            # save to DB
        except Exception as e:
            current_app.logger.debug('Error running model: {}'.format(e))
            return {'error': e}, 500

        return 'made a prediction', 200

    elif request.method == 'DELETE':
        #import pdb; pdb.set_trace()
        print 'delete request received'
        try:
            Experiment.query.filter_by(id=id).delete()
            Sensor.query.filter_by(experiment_id=id).delete()
            current_app.logger.debug('Deleting experiment: {}'.format(id))
            db.session.commit()
        except Exception as e:
            current_app.logger.debug('Error deleting experiment: {}, {}'.format(id, e))
            abort(500)
        if id:
            return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
        else:
            abort(500)
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
