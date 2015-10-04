from flask import Flask, render_template
from werkzeug.exceptions import default_exceptions, HTTPException

from .auth import login_manager
from .data import db
from .sensors.views import sensors
from .users.views import users

import app.errors
import app.logs

app = Flask(__name__)
app.config.from_object('config.DebugConfiguration')


@app.context_processor
def provide_constants():
    return {"constants": {"TUTORIAL_PART": 2}}

db.init_app(app)

login_manager.init_app(app)

app.register_blueprint(sensors)
app.register_blueprint(users)



@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(400)
def key_error(e):
    return render_template('400.html'), 400


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('generic.html'), 500


@app.errorhandler(Exception)
def unhandled_exception(e):
    return render_template('generic.html'), 500


errors.init_app(app)
logs.init_app(app)