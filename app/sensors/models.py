from datetime import datetime

from app.data import CRUDMixin, db


class Experiment(CRUDMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)
    hardware = db.Column(db.Text)
    t_stamp = db.Column(db.DateTime)
    label = db.Column(db.Text)
    sensors = db.relationship('Sensor', backref='experiment', lazy='dynamic')
    
    def __repr__(self):
        return '<Experiment %r>, <id %r>' % (self.hardware, self.id)

class Sensor(CRUDMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)
    SENSOR_TYPE = db.Column(db.Text)
    ACCELEROMETER_X = db.Column(db.Text)
    ACCELEROMETER_Y = db.Column(db.Text)
    ACCELEROMETER_Z = db.Column(db.Text)
    X_AXIS = db.Column(db.Text)
    Y_AXIS = db.Column(db.Text)
    Z_AXIS = db.Column(db.Text)
    LINEAR_ACCELERATION_X = db.Column(db.Text)
    LINEAR_ACCELERATION_Y = db.Column(db.Text)
    LINEAR_ACCELERATION_Z = db.Column(db.Text)
    GYROSCOPE_X = db.Column(db.Text)
    GYROSCOPE_Y = db.Column(db.Text)
    GYROSCOPE_Z =  db.Column(db.Text)
    ORIENTATION_Z = db.Column(db.Text)
    ORIENTATION_X = db.Column(db.Text)
    ORIENTATION_Y = db.Column(db.Text)
    Time_since_start = db.Column(db.Text)
    state = db.Column(db.Text)
    timestamp = db.Column(db.DateTime)
    prediction = db.Column(db.Text)
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiment.id'))

    def __init__(self, experiment, SENSOR_TYPE=None, ACCELEROMETER_X=None,
                ACCELEROMETER_Y=None, ACCELEROMETER_Z=None,
                X_AXIS=None, Y_AXIS=None,
                Z_AXIS=None, LINEAR_ACCELERATION_X=None,
                LINEAR_ACCELERATION_Y=None, LINEAR_ACCELERATION_Z=None,
                GYROSCOPE_X=None, GYROSCOPE_Y=None,
                GYROSCOPE_Z=None,
                ORIENTATION_Z=None, ORIENTATION_X=None,
                ORIENTATION_Y=None,
                Time_since_start=None, state=None,
                timestamp=None, prediction=None
                ):

        self.SENSOR_TYPE = SENSOR_TYPE
        self.ACCELEROMETER_X = ACCELEROMETER_X
        self.ACCELEROMETER_Y = ACCELEROMETER_Y
        self.ACCELEROMETER_Z = ACCELEROMETER_Z
        self.X_AXIS = X_AXIS
        self.Y_AXIS = Y_AXIS
        self.Z_AXIS = Z_AXIS
        self.LINEAR_ACCELERATION_X = LINEAR_ACCELERATION_X
        self.LINEAR_ACCELERATION_Y = LINEAR_ACCELERATION_Y
        self.LINEAR_ACCELERATION_Z = LINEAR_ACCELERATION_Z
        self.GYROSCOPE_X = GYROSCOPE_X
        self.GYROSCOPE_Y = GYROSCOPE_Y
        self.GYROSCOPE_Z = GYROSCOPE_Z
        self.ORIENTATION_Z = ORIENTATION_Z
        self.ORIENTATION_X = ORIENTATION_X
        self.ORIENTATION_Y = ORIENTATION_Y
        self.Time_since_start = Time_since_start
        self.state = state
        self.timestamp = timestamp
        self.prediction = prediction # not in uploaded files
        self.experiment_id = experiment.id # not in uploaded files
    
    def __repr__(self):
        return '<Timestamp {:d}>'.format(self.timestamp)


class Site(CRUDMixin, db.Model):
    __tablename__ = 'tracking_site'

    base_url = db.Column(db.String)
    visits = db.relationship('Visit', backref='site', lazy='select')
    user_id = db.Column(db.Integer, db.ForeignKey('users_user.id'))

    def __repr__(self):
        return '<Site {:d} {}>'.format(self.id, self.base_url)

    def __str__(self):
        return self.base_url


class Visit(CRUDMixin, db.Model):
    __tablename__ = 'tracking_visit'

    browser = db.Column(db.String)
    date = db.Column(db.DateTime)
    event = db.Column(db.String)
    url = db.Column(db.String)
    ip_address = db.Column(db.String)
    location = db.Column(db.String)
    latitude = db.Column(db.Numeric)
    longitude = db.Column(db.Numeric)
    site_id = db.Column(db.Integer, db.ForeignKey('tracking_site.id'))

    def __repr__(self):
        r = '<Visit for site ID {:d}: {} - {:%Y-%m-%d %H:%M:%S}>'
        return r.format(self.site_id, self.url, self.date)
