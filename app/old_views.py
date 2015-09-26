import os
from flask import render_template, flash, redirect, session, url_for, request, g, send_from_directory
from flask.ext.login import login_user, logout_user, current_user, \
    login_required

from app import app
# from .forms import LoginForm
# from .models import User

from werkzeug import secure_filename

ALLOWED_EXTENSIONS = set(['txt', 'csv'])
UPLOAD_FOLDER = os.path.realpath('.')+'/app/uploads'

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS




@app.route('/')
@app.route('/index')
def index():
    posts = [
        {
            'author': {'nickname': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'nickname': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template('index.html',
                           title='Home',
                           posts=posts)


@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html',
                           title='Sign In',
                           form=form,
                           providers=app.config['OPENID_PROVIDERS'])


@app.route('/csv', methods=['GET', 'POST'])
def csv():
    if request.method == 'GET':
        return render_template('csv.html')
    elif request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            # TODO save to DB function
            return redirect(url_for('complete'))
        else:
            print "error"
    else:
        return '404'


@app.route('/csv/<filename>')
def uploaded_file(filename):
    print "here"
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/complete')
def complete():
     return render_template('complete.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))
