import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    out = prediction[0]
    
    if out==0:
        output='Setosa'
    if out==1:
        output='Versicolor'
    if out==2:
        output='Verginica'


    return render_template('index.html', prediction_text='The predicted flower is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)