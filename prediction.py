from imageai.Prediction.Custom import CustomImagePrediction
import os
import json
import collections

def predict(filename):
    execution_path = os.getcwd()

    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("model.h5")
    prediction.setJsonPath("styles.json")
    prediction.loadModel(num_objects=25)

    predictions, probabilities = prediction.predictImage(filename, result_count=3)
    result = collections.defaultdict()
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        result[eachPrediction] = eachProbability
        print(eachPrediction, " : ", eachProbability)
        
    return result