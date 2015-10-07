import requests
import csv
import json

from cStringIO import StringIO
from datetime import datetime

from flask import Request, url_for
from werkzeug import FileStorage
from werkzeug.datastructures import MultiDict

from app.data import CRUDMixin, db
from app.test_base import BaseTestCase
from app.users.models import User

from app.sensors import views
from ..api import views
from app.sensors.models import Sensor


class APITests(BaseTestCase):
    def test_csv_url(self):
        with self.client:
            res = self.client.get('/api/csv')
            self.assert200(res)
            print res.status_code
            print res.data

    

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
