from app import db

class Reading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device = db.Column(db.String(120), index=True, unique=False)
    timestamp = db.Column(db.String(120), index=True, unique=True)
    alt_timestamp = db.Column(db.DateTime)
    acc_x = db.Column(db.Integer, index=True, unique=False)
    acc_y = db.Column(db.Integer, index=True, unique=False)
    acc_z = db.Column(db.Integer, index=True, unique=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    #TODO: time interval

    def __repr__(self):
        return '<Reading %r' % (self.alt_timestamp)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=False)
    email = db.Column(db.String(120), index=True, unique=False)
    # For a one-to-many relationship a db.relationship field is normally defined on the "one" side.
    readings = db.relationship('Reading', backref='user', lazy='dynamic') 

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        try:
            return unicode(self.id)  # python 2
        except NameError:
            return str(self.id)  # python 3

    def __repr__(self):
        return '<User %r' % (self.email)