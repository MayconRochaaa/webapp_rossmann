import os
import pandas as pd
import pickle
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

#load model

BASE_DIR = os.path.abspath('')
MODEL_DIR = os.path.join(BASE_DIR,'model')

model = pickle.load(open(os.path.join(MODEL_DIR,'model_rossman.pkl'), 'rb'))

app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: #there is data
        if isinstance(test_json, dict): #unique example
            teste_raw = pd.DataFrame(test_json, index=[0])
        else: #Multiple example
            teste_raw = pd.DataFrame(test_json, columns= test_json[0].keys())
            
        # instantiate rossmann class
        pipeline = Rossmann()
        
        #data cleaning
        df1 = pipeline.data_cleaning(teste_raw)
        
        #data engineering
        df2 = pipeline.feature_engineering(df1)
        
        #data preparation
        df3 = pipeline.data_preparation(df2)
        
        #prediction
        df_response = pipeline.get_prediction(model, teste_raw, df3)
        
        return df_response
        
        
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port = port)
