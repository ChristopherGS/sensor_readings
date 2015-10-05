from flask import Flask, render_template, request
from werkzeug.exceptions import default_exceptions, HTTPException

from .auth import login_manager
from .data import db
from .sensors.views import sensors
from .sensors import views
from .users.views import users

import app.errors
# import app.logs

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