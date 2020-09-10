# importing libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# app name declare
app = Flask(__name__)
# load model file
model = pickle.load(open('model.pkl', 'rb'))



# @app.something <--we need to do this because to create any number of uri with respect to api

# redirect user to the home page -- root 
@app.route('/')
def home():
    return render_template('index.html')

# predict API
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # as it is a POST request, we will read it from from request.form.values()
    # all of these will be placed into int_features
    int_features = [int(x) for x in request.form.values()]
    
    # then we will convert all of those values into array
    final_features = [np.array(int_features)]
    
    # then we will do the prediction
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


# main function
if __name__ == "__main__":
    app.run(debug=True)