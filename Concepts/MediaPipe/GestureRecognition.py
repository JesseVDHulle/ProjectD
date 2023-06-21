import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, os
import numpy as np


# Path to model, change this to your own path
model_path = "C:\\Users\\Martijn\\Documents\\GitHub\\ProjectD\\Concepts\\MediaPipe\\models\\gesture_recognizer.task"


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a gesture recognizer instance with the live stream mode:
def print_result(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):
    if len(result.gestures) > 0:
        #print(result)
        print(f"{result.gestures[0][0].category_name}")
    else:
        print("No hands detected")


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)


# prepare the webcam
capture_device = cv2.VideoCapture(
    0
)  # <- change this number if you have multiple cameras

# Check if the webcam is opened correctly
if not capture_device.isOpened():
    raise IOError(
        "Cannot open webcam, did you use the right number in cv2.VideoCapture(?)"
    )


with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        _, frame = capture_device.read()

        if not _:
            raise IOError("Cannot read a frame from the webcam")
        numpy_frame_from_opencv = np.array(frame)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv
        )

        frame_timestamp_ms = round(capture_device.get(cv2.CAP_PROP_POS_MSEC))

        recognizer.recognize_async(mp_image, frame_timestamp_ms)
