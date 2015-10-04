from flask import Flask, render_template, current_app, Markup, request
from werkzeug.exceptions import default_exceptions, HTTPException

from .auth import login_manager
from .data import db
from .sensors.views import sensors
from .users.views import users

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


def error_handler(error):
    msg = "Request resulted in {}".format(error)
    current_app.logger.warning(msg, exc_info=error)

    if isinstance(error, HTTPException):
        description = error.get_description(request.environ)
        code = error.code
        name = error.name
    else:
        description = ("We encountered an error "
                       "while trying to fulfill your request")
        code = 500
        name = 'Internal Server Error'

    # Flask supports looking up multiple templates and rendering the first
    # one it finds.  This will let us create specific error pages
    # for errors where we can provide the user some additional help.
    # (Like a 404, for example).
    templates_to_try = ['errors/{}.html'.format(code), 'errors/generic.html']
    return render_template(templates_to_try,
                           code=code,
                           name=Markup(name),
                           description=Markup(description),
                           error=error)


def init_app(app):
    for exception in default_exceptions:
        app.register_error_handler(exception, error_handler)

    app.register_error_handler(Exception, error_handler)

init_app(app)