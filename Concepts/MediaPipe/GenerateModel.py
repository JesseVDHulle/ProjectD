import os
import tensorflow as tf
assert tf.__version__.startswith('2')
from mediapipe_model_maker import gesture_recognizer
import matplotlib.pyplot as plt

dataset_path = "C:\\Users\\Martijn\\Documents\\GitHub\\ProjectD\\Concepts\\MediaPipe\\models\\rps_data_sample"

# Load the dataset
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)

# Split the dataset: 80% training, 10% validation and 10% testing
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# Training the model
hparams = gesture_recognizer.HParams(export_dir="RockPaperScissors_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# Evaluate the model
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

model.export_model()