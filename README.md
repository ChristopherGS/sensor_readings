Sensor Readings
=========

Flask server for receiving accelerometer data from wearable devices

Installation
------------

- pip install -r requirements.txt

- Create an uploads folder inside the "app" directory



Running
-------

To run the application in the development web server just execute `run.py` with the Python interpreter from the flask virtual environment.


Testing
-------

python -m unittest discover

On OSX the lxml package may need to be manually installed due to a bug with that package


TODO
-------

- Download as a CSV function
- Organization for when number of experiments is very high
