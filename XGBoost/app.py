from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the saved XGBoost model
model = pickle.load(open('model_xgboost.pkl', 'rb'))

# Define a function to preprocess the user inputs
def preprocess_inputs(year, employer, job_title, inputs):
    # Convert true/false inputs to 1/0
    for i in range(0, 16):
        if inputs[i] == 'true':
            inputs[i] = 1
        else:
            inputs[i] = 0
    # Convert all inputs to floats
    inputs = [float(i) for i in inputs]
    # Convert the year to an integer
    year = int(year)
    # Encode the inputs as UTF-8
    inputs = [str(i).encode('utf-8') for i in inputs]
    return np.array([year, employer, job_title] + inputs).reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user inputs from the web form
    year = request.form['Year']
    employer = request.form['Employer_Numeric']
    job_title = request.form['Job_Title_Numeric']
    inputs = [x for x in request.form.values()][3:]
    # Preprocess the inputs
    inputs = preprocess_inputs(year, employer, job_title, inputs)
    # Make a prediction using the XGBoost model
    prediction = model.predict(inputs)[0]
    # Format the prediction as a string
    prediction_str = "${:,.2f}".format(prediction)
    # Render the index.html template with the predicted salary and benefits
    return render_template('index.html', prediction=prediction_str)


if __name__ == '__main__':
    app.run(debug=False, port=4500)
