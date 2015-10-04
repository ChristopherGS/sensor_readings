import os
import sqlite3

import numpy as np
import pandas as pd

from app.data import CRUDMixin, db

sensors = Blueprint("sensors", __name__, static_folder='static', template_folder='templates')

DATABASE = 'app.db'

FILES = ['file.csv']

class CSVHelper():
    def csv2sql(conn, files):
