import csv
import json
import os
from datetime import datetime
from random import choice
from time import mktime

import pandas as pd
from flask import (Blueprint, current_app, Markup, Response, abort, flash, jsonify,
                   redirect, render_template, request, session, url_for)
from flask.ext.login import current_user, login_required
from numpy import genfromtxt
from werkzeug import secure_filename
from werkzeug.exceptions import default_exceptions, HTTPException

from app.data import db, query_to_list
from app.science import pandas_cleanup, sql_to_pandas

from .models import Experiment, Sensor, Site, Visit

sensors = Blueprint("sensors", __name__, static_folder='static', template_folder='templates')

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
        current_app.logger.debug('Error changing timestamp {}'.format(e))
        final_timestamp = unknown_timestamp
    return final_timestamp



@sensors.route("/")
@sensors.route("/index")
def index():
    # import pdb; pdb.set_trace()
    return render_template("index.html")



@sensors.route('/csv', methods=['GET', 'POST'])
def csv_route():
    if request.method == 'GET':
        return render_template('sensors/csv.html')
    elif request.method == 'POST':
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
                current_app.logger.debug('Committing {} records to the database'.format(count))
                db.session.commit() #Attempt to commit all the records
                 
            except Exception as e:
                print e
                db.session.rollback() #Rollback the changes on error
                # TODO SEND INFO BACK TO DEVICE
            finally:
                pass
                # db.session.close() #Close the connection
            
            # Test change for Urs
            flash("CSV data saved to DB") 
            return redirect(url_for('sensors.complete')), 201
        else:
            return render_template('index.html'), 400
    else:
        return '404'

@sensors.route('/complete')
def complete():
     return render_template('sensors/complete.html')

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

@sensors.route('/display/<int:id>')
def display_id(id):
    query = db.session.query(Sensor)
    df = pd.read_sql_query(query.statement, query.session.bind)
    pandas_id = id
    df2 = df[df.experiment_id == pandas_id]
    db_index_choice = df2
    experiment_number = pd.unique(df2.experiment_id.values)
    return render_template('sensors/file_details.html', experiment_number=experiment_number[0], 
        db_index_choice=db_index_choice.to_html(), id=id)

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

    print df

    d3_json = df.to_json(orient='records')

    d3_response = json.dumps(d3_json)


    return render_template('sensors/file_graph.html', experiment_number=experiment_number[0], 
        db_index_choice=db_index_choice.to_html(), id=id, d3_response = d3_response)




