import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('IrisPredictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    finalFeatures = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
#     species = list(model.predict(features))[0]
#     msg = 'Expected Species is : ' + species
    
    prediction = model.predict(finalFeatures)

#     return render_template('index.html', prediction_text=msg)
#     return render_template('index.html', prediction_text='Predicted Results: {}'.format(prediction))
    return render_template('index.html', prediction_text='Predicted Results: {}'.format(prediction[0]))

#     return render_template('index.html', prediction_text=prediction[0])



if __name__ == "__main__":
    app.run(debug=True)