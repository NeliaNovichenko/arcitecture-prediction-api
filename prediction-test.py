from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "./dataset/models/try2_model_ex-006_acc-0.714195.h5"))
prediction.setJsonPath(os.path.join(execution_path, "./dataset/json/model_class.json"))
prediction.loadModel(num_objects=25)

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "./dataset/train/queenAnne/144_800px-Isbister_School.jpg"), result_count=25)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : " , eachProbability)