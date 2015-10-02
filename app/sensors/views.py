import os, csv
from datetime import datetime
from numpy import genfromtxt
import pandas as pd

from flask import abort, Blueprint, flash, jsonify, Markup, redirect, render_template, request, \
url_for, session
from random import choice
from flask.ext.login import current_user, login_required

from .forms import SiteForm, VisitForm
from .models import Site, Visit, Sensor
from app.data import query_to_list, db
from app.science import sql_to_pandas

from werkzeug import secure_filename


sensors = Blueprint("sensors", __name__, static_folder='static', template_folder='templates')


ALLOWED_EXTENSIONS = set(['txt', 'csv'])
UPLOAD_FOLDER = os.path.realpath('.')+'/app/uploads'

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def Load_Data(file_name):
    # data = genfromtxt(file_name, delimiter=',', skiprows=1)
    data = pd.read_table(file_name, header=None, skiprows=1, delimiter=',') 
    return data.values.tolist()



@sensors.route("/")
@sensors.route("/index")
def index():
    # site_form = SiteForm()
    # visit_form = VisitForm()
    return render_template("index.html")



@sensors.route('/csv', methods=['GET', 'POST'])
def csv_route():
    if request.method == 'GET':
        return render_template('sensors/csv.html')
    elif request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            # need to open the CSV and iterate row by row
            date = datetime.now()

            try:
                print filename
                file_name = filename
                data = Load_Data(os.path.join(UPLOAD_FOLDER, file_name))
                count = 0 
                holder = []
                for i in data:
                    print type(i[3])
                    my_timestamp = datetime.strptime(i[3], "%Y-%m-%d %H:%M:%S:%f")
                    record = Sensor(**{
                        'accelerometer_x' : i[0],
                        'accelerometer_y' : i[1],
                        'accelerometer_z' : i[2],
                        'device' : 'Nexus 5',
                        'timestamp' : my_timestamp

                    })
                    
                    db.session.add(record) #Add all the records
                    count += 1
                    
                if ((count % 10 == 0) | (count < 20)):
                    db.session.commit() #Attempt to commit all the records
                 
                print holder
            except Exception as e:
                print e
                db.session.rollback() #Rollback the changes on error
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

@sensors.route('/display')
def display():
    sql_to_pandas()
    names = os.listdir(UPLOAD_FOLDER)
    print type(names)
    file_url = url_for('static', filename=os.path.join('uploads', choice(names)))
    return render_template('sensors/show_files.html', file_url=names)


"""
@tracking.route("/site", methods=("POST", ))
def add_site():
    form = SiteForm()
    if form.validate_on_submit():
        site = Site()
        form.populate_obj(site)
        db.session.add(site)
        db.session.commit()
        flash("Added site")
        return redirect(url_for(".index"))

    return render_template("validation_error.html", form=form)


@tracking.route("/site/<int:site_id>")
def view_site_visits(site_id=None):
    site = Site.query.get_or_404(site_id)
    query = Visit.query.filter(Visit.site_id == site_id)
    data = query_to_list(query)
    title = "visits for {}".format(site.base_url)
    return render_template("data_list.html", data=data, title=title)


@tracking.route("/visit", methods=("POST", ))
@tracking.route("/site/<int:site_id>/visit", methods=("POST",))
def add_visit(site_id=None):
    if site_id is None:
        # This is only used by the visit_form on the index page.
        form = VisitForm()
    else:
        site = Site.query.get_or_404(site_id)
        # WTForms does not coerce obj or keyword arguments
        # (otherwise, we could just pass in `site=site_id`)
        # CSRF is disabled in this case because we will *want*
        # users to be able to hit the /site/:id endpoint from other sites.
        form = VisitForm(csrf_enabled=False, site=site)

    if form.validate_on_submit():
        visit = Visit()
        form.populate_obj(visit)
        visit.site_id = form.site.data.id
        db.session.add(visit)
        db.session.commit()
        flash("Added visit for site {}".format(form.site.data.base_url))
        return redirect(url_for(".index"))

    return render_template("validation_error.html", form=form)


@tracking.route("/sites")
def view_sites():
    query = Site.query.filter(Site.id >= 0)
    data = query_to_list(query)

    # The header row should not be linked
    results = [next(data)]
    for row in data:
        row = [_make_link(cell) if i == 0 else cell
               for i, cell in enumerate(row)]
        results.append(row)

    return render_template("data_list.html", data=results, title="Sites")


_LINK = Markup('<a href="{url}">{name}</a>')


def _make_link(site_id):
    url = url_for(".view_site_visits", site_id=site_id)
    return _LINK.format(url=url, name=site_id)


"""