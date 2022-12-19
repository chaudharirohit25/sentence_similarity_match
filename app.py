from flask import Flask,render_template,request
from query import *
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import os

print("inside app.py")
app = Flask(__name__)
print("flask app created.")
@app.route('/', methods = ['POST', 'GET'])
def form():
    print("inside form")
    full_filename = '/static/IMG/CDAC_logo.jpeg'
    return render_template('form.html')
print("form function created.")
@app.route('/data', methods = ['POST', 'GET'])
def data():
    print("inside data.")
    full_filename = '/static/IMG/CDAC_logo.jpeg'
    form_data = request.form
    preds = query(form_data)
    return render_template('data.html',form_data = form_data,
                             pred_value=preds)
print("data function created.")

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
print("In main function")
