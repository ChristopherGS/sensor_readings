from datetime import datetime

from app.data import CRUDMixin, db

class Experiment(CRUDMixin, db.Model):
    #  __tablename__ = 'tracking_experiment'

    id = db.Column(db.Integer, primary_key=True)
    hardware = db.Column(db.String(120))
    sensors = db.relationship('Sensor', backref='experiment', lazy='dynamic')
    

    def __init__(self, hardware="unknown"):
        pass
        """
        my_date = datetime.now()
        self.t_stamp = my_date
        self.hardware = hardware
        """
    def __repr__(self):
        return '<Timestamp {:d}>'.format(self.t_stamp)

class Sensor(CRUDMixin, db.Model):
    #__tablename__ = 'tracking_sensor'
    id = db.Column(db.Integer, primary_key=True)
    accelerometer_x = db.Column(db.Text)
    accelerometer_y = db.Column(db.Text)
    accelerometer_z = db.Column(db.Text)
    timestamp = db.Column(db.DateTime)
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiment.id'))
    

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