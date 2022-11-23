from flask import Flask,render_template,request
from query import *
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def form():
    full_filename = '/opt/static/IMG/CDAC_logo.jpeg'
    return render_template('form.html')

@app.route('/data', methods = ['POST', 'GET'])
def data():
    full_filename = '/opt/static/IMG/CDAC_logo.jpeg'
    form_data = request.form
    preds = query(form_data)
    return render_template('data.html',form_data = form_data,
                             pred_value=preds)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
