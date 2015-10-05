from datetime import datetime

from flask.ext.testing import TestCase

from app.sensors.models import Sensor

from . import app, db


class BaseTestCase(TestCase):
    """A base test case for flask-tracking."""

    def create_app(self):
        app.config.from_object('config.TestConfiguration')
        return app

    def setUp(self):
        db.create_all()
        date_str = "2008-11-10 17:53:59:400"
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S:%f")

        sensor = Sensor(
            accelerometer_x = '9.10', 
            accelerometer_y = '2.20',
            accelerometer_z = '3.40',
            timestamp = dt_obj
            # device = 'Nexus 5'
        )

        db.session.add(sensor)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
