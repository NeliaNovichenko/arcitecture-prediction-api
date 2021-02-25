from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet50()
prediction.setModelPath("model-old.h5")
prediction.setJsonPath("styles.json")
prediction.loadModel(num_objects=25)

predictions, probabilities = prediction.predictImage("test\ann.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)