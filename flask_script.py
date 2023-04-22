import joblib
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

 
@app.route("/predict", methods=["POST", "GET"])
def predict():

    model_name, data = request.get_json()["data"]


        
    if(model_name == 'Heuristic'):
        model = joblib.load("./models/dummy_best.joblib")
        prediction = model.predict(data) 
    
    if(model_name == 'DecisionTree'):
        model = joblib.load("./models/decision_tree_best.joblib")
        prediction = model.predict(data)

    if(model_name == 'RandomForest'):
        model = joblib.load("./models/random_forest_best.joblib")
        prediction = model.predict(data)   

  
    if(model_name == "NeuralNetwork"):
        model = load_model('./models/neural_network_best.h5')
        prediction = model.predict(data)
        prediction = np.argmax(prediction)        
  
   
    return jsonify({"Predicted class": prediction.tolist(), "Model": model_name})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
