Sensor Readings
=========

Flask server for receiving accelerometer and gyroscope timestamp data from metawear boards.

Installation
------------

- pip install -r requirements.txt

- Create an uploads folder inside the "app" directory

- install the hmmlearn module: https://github.com/hmmlearn/hmmlearn



Running
-------

To run the application in the development web server just execute `run.py` with the Python interpreter from the flask virtual environment.

When deploying to a server, you will need to create an uploads folder in app/api/uploads


Testing
-------
No tests available yet

python -m unittest discover

On OSX the lxml package may need to be manually installed due to a bug with that package


TODO
-------
- Indexing for when number of experiments is very high
