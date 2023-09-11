from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras_preprocessing.sequence import pad_sequences
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras.models
from keras.models import model_from_json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,flash
from werkzeug.utils import secure_filename
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model, Sequential, load_model
from tensorflow.python.keras.models import Input
import pickle
import h5py
import re
# create Flask application
from flask_mysqldb import MySQL
import MySQLdb.cursors
app = Flask(__name__) #Initialize the flask App
app.secret_key = 'your secret key'
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'fakenews'
 
mysql = MySQL(app)

# read object TfidfVectorizer and model from disk
MODEL_PATH ='cnn.h5'
model = load_model(MODEL_PATH)
 
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')
@app.route('/login') 
def login():
	return render_template('login.html')    
    
@app.route('/loginaction', methods =['GET', 'POST'])
def loginaction():
    
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['pwd']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM register WHERE username = % s AND password = % s', (username, password, ))
        account = cursor.fetchone()
        if account:
            return render_template('upload.html')
        else:
            return 'Invalid Login'
 
 

@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

 
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/register',methods= ['GET',"POST"])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form and 'age' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        age = request.form['age']
        
        reg = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{6,10}$"
        pattern = re.compile(reg)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # Check if account exists using MySQL)
        cursor.execute('SELECT * FROM register WHERE Username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not re.search(pattern,password):
            msg = 'Password should contain atleast one number, one lower case character, one uppercase character,one special symbol and must be between 6 to 10 characters long'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into employee table
            cursor.execute('INSERT INTO register VALUES (NULL, %s, %s, %s, %s)', (username, password, email, age))
            mysql.connection.commit()
            flash('You have successfully registered! Please proceed for login!')
            return redirect(url_for('login'))
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
        return msg
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)



@app.route('/predict', methods=['POST'])
def predict():
    error = None
    if request.method == 'POST':
        # message
        msg = request.form['message']
        msg = pd.DataFrame(index=[0], data=msg, columns=['data'])

        # transform data
        new_text =pad_sequences((tokenizer.texts_to_sequences(msg['data'].astype('U'))), maxlen=547)
          
        # model
        result = model.predict(new_text,batch_size=1,verbose=2)
         
        if result >0.5:
            result = 'Fake'
        else:
            result = 'Real'

        return render_template('index.html', prediction_value=result)
    else:
        error = "Invalid message"
        return render_template('index.html', error=error)
@app.route('/chart') 
def chart():
	return render_template('chart.html')

if __name__ == "__main__":
    app.run(debug=True)
