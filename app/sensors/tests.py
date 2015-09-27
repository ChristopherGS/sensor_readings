from flask import url_for
from flask import Request
from werkzeug import FileStorage
from werkzeug.datastructures import MultiDict
from cStringIO import StringIO

from app.test_base import BaseTestCase
from app.users.models import User
from .models import Site, Visit
from ..sensors import views



class SensorViewsTests(BaseTestCase):
    def test_csv_url(self):
        with self.client:
            res = self.client.get(url_for("sensors.csv"))
            self.assert200(res)

    def test_complete_url(self):
        with self.client:
            res = self.client.get(url_for("sensors.complete"))
            self.assert200(res)

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
                """A testing request to use that will return a
                TestingFileStorage to test the uploading."""
                @property
                def files(self):
                    d = MultiDict()
                    d['file'] = TestingFileStorage(filename=filename)
                    return d

            self.app.request_class = TestingRequest
            test_client = self.app.test_client()
            rv = test_client.post(
                url_for("sensors.csv"),
                data=dict(
                    file=(StringIO('Foo bar baz'), filename),
                ))
            self.assertEqual(rv.status_code, status_code)


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




    """
        file = open('file.csv','r')
        with self.client:
            res = self.client.post(url_for("sensors.csv"), {'file': file})
            print res.status_code
            print res.data
            # self.assert200(res)
    """
     #   file = open('file.csv','r') 
     #   res = self.client.post('/csv', {'file': file})

     #res = self.client.post(url_for("sensors.csv"), data=dict(
     #          upload_var=(StringIO("yo"), 'file.csv'),
     #      ))
        
      #  print res.status_code
       # print res.data
        # assert res.status_code == 200
        # assert 'file saved' in res.data


if __name__ == '__main__':
    unittest.main()