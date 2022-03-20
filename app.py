from flask import Flask, render_template, request
import json
import os
import urllib.request
import numpy as np
import pathlib
from fastai.text.all import *;

app = Flask(__name__)

learner = load_learner('stance_prediction_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html') 
    # data = request.form.get('data', '')  # pass the form field name as key
    # return data

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.form.get('data', '')  # pass the form field name as key
    prediction = learner.predict(data)[0]
    ans = ''
    if prediction == 'A':
        ans = "Against"
    elif prediction == 'F':
        ans = "For"
    else:
        ans = "Neutral"
    return render_template('prediction.html', data=ans)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
