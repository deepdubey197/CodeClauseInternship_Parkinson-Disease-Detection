import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_pymongo import PyMongo
model=joblib.load('model.pkl')
scaler=joblib.load('scaler.pkl')
app=Flask(__name__)

app.config['MONGO_URI'] = 'mongodb://localhost:27017/Parkinson'  # Replace with your MongoDB URI
mongo = PyMongo(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_data=[float(x) for x in request.form.values()]
    input_data_array=np.asarray(input_data)
    input_data_reshaped=input_data_array.reshape(1,-1)
    std_data=scaler.transform(input_data_reshaped)
    prediction=model.predict(std_data)

    if prediction[0]==0:
        result= 'does not have'
    else:
        result= 'has'

    prediction_result = int(prediction[0])

    mongo.db.predictions.insert_one({
        'input_data': input_data,
        'prediction_result': prediction_result,
        'prediction_text': result
    })

    return render_template('index.html',prediction_text=result)

if __name__=='__main__':
    app.run(debug=True)