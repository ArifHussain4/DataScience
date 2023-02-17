from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv("trainingdata.csv")

# Load the saved XGBoost model
model = pickle.load(open('model_xgboost.pkl', 'rb'))

# Load the saved LabelEncoder objects
with open('employer_le.pkl', 'rb') as f:
    employer_le = pickle.load(f)
    
with open('job_title_le.pkl', 'rb') as f:
    job_title_le = pickle.load(f)

# Define a function to preprocess the user inputs
def preprocess_inputs(employer, job_title, inputs):
    # Check that inputs has at least 19 elements
    # Convert true/false inputs to 1/0
    for i in range(1, 17):
        if inputs[i] == 'true':
            inputs[i] = 1
        else:
            inputs[i] = 0
    # Convert employer and job title to numerical values
    employer_numeric = employer_le.transform([employer])
    job_title_numeric = job_title_le.transform([job_title]) 
    # Delete employer and job title from inputs
    del inputs[-2:] 
    # Add numerical values for employer and job title to input data
    inputs.append(employer_numeric[0])
    inputs.append(job_title_numeric[0])  
    # Convert all inputs to floats
    inputs = [float(i) for i in inputs]
    # Add hardcoded value for Year
    return np.array(inputs).reshape(1, -1)


app = Flask(__name__)

@app.route('/')
def index():
    job_title_list = data['Job Title'].unique().tolist()
    employer_list = data['Employer'].unique().tolist()
    return render_template('index.html', employer=employer_list, Job_Title=job_title_list)
    

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user inputs from the web form
    inputs = [x for x in request.form.values()]
    employer = inputs[-2]
    job_title = inputs[-1]
    inputs = inputs[:-2]
    # Preprocess the inputs
    inputs = preprocess_inputs(employer, job_title, inputs)
    # Make a prediction using the XGBoost model
    prediction = model.predict(inputs)
    # Convert the prediction from an array to a list and format as a string
    prediction = prediction.tolist()[0]
    prediction_str = "${:,.2f}".format(prediction)
    # Render the index.html template with the predicted salary and benefits
    return render_template('index.html', prediction=prediction_str)

if __name__ == '__main__':
    app.run(debug=True, port=5030)
