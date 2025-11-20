from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from templates import index
### WSGI Application
application = Flask(__name__)

app = application

## Route for a home page. This decorator defines how many url 's us that when 
# we go to that particular page what function to do automatically.
@app.route('/')
def index():
    return render_template('index.html') # We take the html from "templates" module

# Result checker html page
@app.route('/predictdata', methods = ['GET','POST'])  # The predictdata is a variable rule. 
def predict_datapoint():
    