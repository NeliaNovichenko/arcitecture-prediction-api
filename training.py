# from imageai.Prediction.Custom import ModelTraining
# # from customClassification import ClassificationModelTrainer

# model_trainer = ModelTraining()
# model_trainer.setModelTypeAsResNet()
# model_trainer.setDataDirectory("dataset")
# model_trainer.trainModel(num_objects=25, num_experiments=25, enhance_data=True, batch_size=32, show_network_summary=True)

# from customClassification import ClassificationModelTrainer

from imageai.Classification.Custom import ClassificationModelTrainer
 
model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory("dataset")
model_trainer.trainModel(num_objects=10, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)
