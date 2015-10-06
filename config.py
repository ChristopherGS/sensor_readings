import os

_basedir = os.path.abspath(os.path.dirname(__file__))

class BaseConfiguration(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'flask-session-insecure-secret-key'

    DATABASE = 'app.db'
    DATABASE_PATH = os.path.join(_basedir, DATABASE)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + DATABASE_PATH
    
    SQLALCHEMY_ECHO = False
    HASH_ROUNDS = 100000
    UPLOAD_FOLDER = os.path.join(_basedir, '/app/uploads')
    _static_folder = os.path.join(_basedir, 'app/static')


    ERROR_LOG_PATH = os.path.join(_basedir, 'logs.txt')

class DebugConfiguration(BaseConfiguration):
    DEBUG = True
    # disable csrf checking for the /csv POST route
    WTF_CSRF_ENABLED = False
    WTF_CSRF_CHECK_DEFAULT = False

class TestConfiguration(BaseConfiguration):
    TESTING = True
    WTF_CSRF_ENABLED = False

    DATABASE = 'tests.db'
    DATABASE_PATH = os.path.join(_basedir, DATABASE)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # + DATABASE_PATH

    # Since we want our unit tests to run quickly
    # we turn this down - the hashing is still done
    # but the time-consuming part is left out.
    HASH_ROUNDS = 1


