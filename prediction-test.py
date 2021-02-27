from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "model_ex-010_acc-0.733146.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=4)

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "697_800px-H._V._Shaw_Building%2C_Edmonton_%281%29.jpg"), result_count=4)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : " , eachProbability)