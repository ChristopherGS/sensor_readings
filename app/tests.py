import nose
import unittest
from app import app
from cStringIO import StringIO


class BasicTestCase(unittest.TestCase):
    
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/csv', content_type='html/text')
        # self.assertEqual(response.status_code, 200)
        # self.assertEqual(response.data, "Hello World")
        print response.data

"""
class UploadTest(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_upload(self):
        res = self.client.post('/csv', data=dict(
            upload_var=(StringIO("yo"), 'test.txt'),
        ))
        print res.status_code
        assert res.status_code == 302
        # assert 'file saved' in res.data

def test2():
    pass

"""

if __name__ == '__main__':
    unittest.main()