Sensor Readings (WIP)
=====================

Flask server for receiving accelerometer and gyroscope timestamp data from metawear boards.

Uses sklearn for machine learning tasks, data is serialized on the server with joblib. The first time you run the server
it take time as the data is trained and serialized. 

A sandbox is provided which works without the rest of the flask application - easiest for quick experimentation


Installation
------------

- pip install -r requirements.txt (TODO: some are missing)

- Create a "pickle" folder in the root directory to store the .pkl serialized data 

- Create an uploads folder inside the "app" directory

- (optional) Install keras for RNN work (GPU is recommended for speed)


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
- Upload more training data
- Enable easy algorithm switching
- RNN implementation
- Indexing for when number of experiments is very high
- Optimize speed from Android upload (probably some kind of chunking)
