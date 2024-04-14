# app.py
from flask import Flask

app = Flask(__name__)

# Import routes after initializing the Flask app to avoid circular imports
from routes import *

if __name__ == "__main__":
    app.run(debug=True)
