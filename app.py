from flask import Flask,request
import pandas as pd
from _collections import OrderedDict
import joblib

app=Flask(__name__)


@app.route('/api/risk')
def get():
    male=float(request.args['Male'])
    age=float(request.args['Age'])
    smoking=float(request.args['Smoking'])
    chol=float(request.args['Chol'])
    bp=float(request.args['BP'])
    hrate=float(request.args['Hrate'])
    glue=float(request.args['Glue'])
    alcohol=float(request.args['Alcohol'])

    outFileFolder = 'RiskPredictionModel/'
    filePath = outFileFolder + 'risk_prediction_model.joblib'
    file = open(filePath, "rb")
    trained_model = joblib.load(file)

    new_data=OrderedDict([('male',male),('age',age),('alcohol',alcohol),('cigsPerDay',smoking),('totChol',chol),('diaBP',bp),('heartRate',hrate),('glucose',glue)])
    new_data=pd.Series(new_data).values.reshape(1,-1)
    prediction = trained_model.predict(new_data)
    return str(prediction)

if __name__ == '__main__':
    app.run()
