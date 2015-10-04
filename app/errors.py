from flask import Flask, render_template, current_app, Markup, request, current_app
from werkzeug.exceptions import default_exceptions, HTTPException

def error_handler(error):
    error_name = error
    current_app.logger.warning('Request resulted in {}'.format(error), exc_info=error)
    # ... etc. ...

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