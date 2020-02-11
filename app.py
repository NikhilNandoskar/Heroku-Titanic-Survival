import numpy as np
import flask
from flask import render_template
from Titanic_Dataset import Output_classes,passenger_id, sc
from LR_Regularization_Dropout_Adam import api_prediction
import pickle

app = flask.Flask(__name__, template_folder='templates')
model_to_pickle = "Logisitic_Regression.pkl"
with open(model_to_pickle, 'rb') as file:
    pickled_model = pickle.load(file)


@app.route('/',methods=['GET','POST'])
def main():
    '''
    For rendering results on HTML GUI
    '''
    if flask.request.method == 'GET':
        return (render_template('index.html'))
    
    if flask.request.method == 'POST':
        test_features = [float(x) for x in flask.request.form.values()]
        final_features = np.array([test_features[1:]])
        final_features = sc.transform(final_features)
        #print(final_features.shape)
        final_features = final_features.reshape((1,-1))
    
        
        prediction = api_prediction(final_features,pickled_model, passenger_id,
                      Output_classes,keep_prob=1,predict_result=True, 
                      activation_type="binary" ,flags="predict_y")
    
        prediction = np.squeeze(prediction).astype(int)
        #print(prediction)
        if prediction==0: pred="Didn't Survive :/"
        else: pred="Survived! :D"
        #output = round(prediction[0], 2)
    
        return render_template('index.html', prediction_text='Did the Passenger Survive? {}'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)