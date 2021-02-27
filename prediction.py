from imageai.Prediction.Custom import CustomImagePrediction
import os
import json
import collections

prediction = None

def getModel():
    global  prediction
    if prediction is not None:
        return prediction
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet50()
    prediction.setModelPath("model.h5")
    prediction.setJsonPath("styles.json")
    prediction.loadModel(num_objects=25)
    return prediction

def predict(filename):
    execution_path = os.getcwd()

    model = getModel()
   
    predictions, probabilities = model.classifyImage(filename, result_count=3)
    result = collections.defaultdict()
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        result[eachPrediction] = eachProbability
        print(eachPrediction, " : ", eachProbability)
        
    return result