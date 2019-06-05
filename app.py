import os

import pandas as pd
import numpy as np

from flask import Flask, jsonify, render_template, request

from predictPrice import predictPrice

app = Flask(__name__)


#################################################
# Global Variables
#################################################
filter_by = {"yr" : 0, "cat" : all, "dist" :0}
samplefile = "InputSamples.csv"
datafilepath = "dataforfinalproject"

# read sample csv
sampdf = pd.read_csv(os.path.join(datafilepath, samplefile), low_memory = False)


#################################################
# Routes setup
#################################################
@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index.html")

# IMPORTANT - PREVENT CACHING BY FLASK
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r



# Filter routes
@app.route("/filterData")
def filterData():
    """Return a json that can be used to populate filters."""
    filt_dict = {} 

    # convert number of rows into dict with each sample number
    filt_dict['Sample'] = [f"Sample {i+1}" for i in range(sampdf.shape[0])]

    
    return jsonify(filt_dict)

# Filter routes
@app.route("/predict/<regVal>/<sqftVal>")
def predPrice(regVal, sqftVal):
    """Return a json that provides details of the metadata chosen and predicted price."""
    
    resultsJSON = {}
    sampdf.rename(columns = {"index":"SAMPLE #"}, inplace = True)
    resultsJSON['metadata'] = sampdf.to_html(table_id = "sampMetaData", classes = "table table-striped table-bordered table-sm")
    resultsJSON['PredictResults'] = predictPrice(int(regVal),int(sqftVal))

    return jsonify(resultsJSON)

if __name__ == "__main__":
    app.run()
