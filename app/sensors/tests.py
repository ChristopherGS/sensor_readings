from cStringIO import StringIO
from datetime import datetime

from flask import Request, url_for
from werkzeug import FileStorage
from werkzeug.datastructures import MultiDict

from app.data import CRUDMixin, db
from app.test_base import BaseTestCase
from app.users.models import User

from ..sensors import views
from .models import Sensor, Site, Visit


class SensorViewsTests(BaseTestCase):
    def test_csv_url(self):
        with self.client:
            res = self.client.get(url_for('sensors.csv_route'))
            self.assert200(res)

    def test_complete_url(self):
        with self.client:
            res = self.client.get(url_for('sensors.complete'))
            self.assert200(res)

    """
    def test_csv_upload(self):
        # Loop over some files and the status codes that we are expecting
        for filename, status_code in \
                (('foo.csv', 201), ('foo.txt', 201), ('foo.pdf', 400),
                 ('foo.py', 400), ('foo', 400)):

            # The reason why we are defining it in here and not outside
            # this method is that we are setting the filename of the
            # TestingFileStorage to be the one in the for loop. This way
            # we can ensure that the filename that we are "uploading"
            # is the same as the one being used by the application
            class TestingRequest(Request):

                #A testing request to use that will return a
                #TestingFileStorage to test the uploading.
                @property
                def files(self):
                    d = MultiDict()
                    d['file'] = TestingFileStorage(filename=filename)
                    return d

            self.app.request_class = TestingRequest
            test_client = self.app.test_client()
            rv = test_client.post(
                url_for("sensors.csv_route"),
                data=dict(
                    file=(StringIO('Foo bar baz'), filename),
                ))
            self.assertEqual(rv.status_code, status_code)

    """

    def test_get_sensor_input(self):
        """Can we retrieve the Sensor instance created in setUp?"""
        with self.client:
            # sensor = Sensor.query.get(107)
            sensorz = Sensor.query.get(1)
            assert sensorz is not None
            self.assertEquals(sensorz.accelerometer_x, '9.10')

    def test_csv2sql(self):
        with self.client:

            date_str = "2008-11-10 17:53:59:400"
            dt_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S:%f")

            sensor = Sensor(
                accelerometer_x = '9.10', 
                accelerometer_y = '2.20',
                accelerometer_z = '3.40',
                timestamp = dt_obj
                #device = 'Nexus 5'
            )

            db.session.add(sensor)
            db.session.commit()

            sensors = Sensor.query.all()

            self.assertEquals('9.10', sensors[0].accelerometer_x)
            self.assertTrue(sensors[0].id > 0)

            # self.assertTrue(type(sensors[0].accelerometer_x) == str)


    def test_sql_demicals(self):
        pass
        # TODO: some decimal checks

    def test_show_file_url(self):
        pass
        # TODO: file checks


    # TODO: check for duplicate files

    def test_file_post(self):
        data = """
        Source,video1393x2352_high,audiowefxwrwf_low,default2325_none,23234_audio,complete_crap,AUDIO_upper_case_test"""

        test_client = self.app.test_client()
        rv = test_client.post('/csv', data=dict(
                                   file=(StringIO(data), 'test.csv'),
                               ), follow_redirects=True)

        print rv
        """
        The origin server MUST create the resource before returning the 201 status code. 
        If the action cannot be carried out immediately, the server SHOULD respond with 202 (Accepted) response instead.
        """
        self.assertEqual(rv.status_code, 201)


class TestingFileStorage(FileStorage):
    """
    This is a helper for testing upload behavior in your application. You
    can manually create it, and its save method is overloaded to set `saved`
    to the name of the file it was saved to. All of these parameters are
    optional, so only bother setting the ones relevant to your application.

    This was copied from Flask-Uploads.

    :param stream: A stream. The default is an empty stream.
    :param filename: The filename uploaded from the client. The default is the
                     stream's name.
    :param name: The name of the form field it was loaded from. The default is
                 ``None``.
    :param content_type: The content type it was uploaded as. The default is
                         ``application/octet-stream``.
    :param content_length: How long it is. The default is -1.
    :param headers: Multipart headers as a `werkzeug.Headers`. The default is
                    ``None``.
    """
    def __init__(self, stream=None, filename=None, name=None,
                 content_type='application/octet-stream', content_length=-1,
                 headers=None):
        FileStorage.__init__(
            self, stream, filename, name=name,
            content_type=content_type, content_length=content_length,
            headers=None)
        self.saved = None

    def save(self, dst, buffer_size=16384):
        """
        This marks the file as saved by setting the `saved` attribute to the
        name of the file it was saved to.

        :param dst: The file to save to.
        :param buffer_size: Ignored.
        """
        if isinstance(dst, basestring):
            self.saved = dst
        else:
            self.saved = dst.name




if __name__ == '__main__':
    unittest.main()
