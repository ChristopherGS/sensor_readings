from app import app

import nose
import unittest

from cStringIO import StringIO


class BasicTestCase(unittest.TestCase):

    def setUp(self):
        # creates a test client
        self.app = app.test_client()
        # propagate the exceptions to the test client
        self.app.testing = True 
    
    def test_index(self):
        # sends HTTP GET request to the application
        # on the specified path
        result = self.app.get('/') 

        # assert the status code of the response
        self.assertEqual(result.status_code, 200) 


class UploadTest(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_csv_url(self):
        res = self.client.get('/csv')
        assert res.status_code == 200

    def test_upload(self):
        # res = self.client.post('/csv', data=dict(
        #    upload_var=(StringIO("yo"), 'file.csv'),
        #))
        #print res.status_code
        #print res.data
        file = open('file.csv','r') 
        res = self.client.post('/csv', {'file': file})
        
        print res.status_code
        print res.data
        # assert res.status_code == 200
        # assert 'file saved' in res.data


if __name__ == '__main__':
    unittest.main()