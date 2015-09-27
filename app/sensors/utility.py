import os

from app.data import CRUDMixin, db

import sqlite3
import pandas as pd
import numpy as np


sensors = Blueprint("sensors", __name__, static_folder='static', template_folder='templates')

DATABASE = 'app.db'

FILES = ['file.csv']

class CSVHelper():
    def csv2sql(conn, files):