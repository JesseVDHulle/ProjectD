import cv2
import matplotlib
import numpy as np
import os

import tensorflow
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import keyboard


def detection(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # color conversion
    img.flags.writeable = False
    results = model.process(img)  # predict
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # revert color order change
    return img, results


def draw_landmarks(img, results):
    mp_draw.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# TODO Check if you can hide certain points where there is a overlap between models
def draw_landmarks_styled(img, results):
    mp_draw.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                           mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                           mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=4),
                           mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_draw.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=4),
                           mp_draw.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2))
    mp_draw.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=4),
                           mp_draw.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2))


# print version of imports
print("CV2 Version: ", cv2.__version__)
print("Numpy Version: ", np.__version__)
print("Matplotlib Version: ", matplotlib.__version__)
print("Mediapipe Version: ", mp.__version__)
print("Tensorflow Keras Version: ", keras.__version__)


# Get and concatenate all landmarks from all models
# TODO possibly remove unnecessary landmarks
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


colors = [(245, 117, 160), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
pTime = 0

mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# DataCollection
DATA_PATH = os.path.join('MEDIAPIPE_DATA')

# Signs we can detect
actions = np.array(['hallo', 'mijn naam is','martijn'])


# thirty videos of data
no_sequences = 1

# Length of the videos in frames
sequence_length = 30

READMODE = False
was_pressed = False
# Model Creation
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
if READMODE:
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):

                    ret, frame = cap.read()
                    # detect
                    img, results = detection(frame, holistic)

                    # print(results)

                    # Draw
                    draw_landmarks_styled(img, results)
                    # New apply wait logic
                    if frame_num == 0:
                        cv2.putText(img, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4,
                                    cv2.LINE_AA)
                        cv2.putText(img, 'Collectiong frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # show to screen
                        cv2.imshow('GestureTrackingTest', img)
                        cv2.waitKey(1000)
                    else:
                        cv2.putText(img, 'Collectiong frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # show to screen
                        cv2.imshow('GestureTrackingTest', img)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    # exit code
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break

        #cap.release()
        # Create labels
        label_map = {label:num for num, label in enumerate(actions)}
        sequences, labels = [], []
        for action in actions:
            for sequence in range(no_sequences):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
                print( np.array(sequences).shape)
       # print(np.asarray(sequences))
        X = np.asarray(sequences)
        y = to_categorical(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

        # LSTM Neural Network
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
        model.summary()
        model.save('action.h5')

        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()

        multilabel_confusion_matrix(ytrue, yhat)
        print(accuracy_score(ytrue, yhat))
else:
    model.load_weights('action.h5')

    # cv2.destroyAllWindows()
    sequence = []
    sentence = []
    threshold = 0.85
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            # detect
            img, results = detection(frame, holistic)

            # print(results)

            # Draw
            draw_landmarks_styled(img, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
            #    res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #    print(actions[np.argmax(res)])
                if keyboard.is_pressed('z'):
                    if not was_pressed:
                        sentence.append(actions[0])
                        was_pressed = True
                elif keyboard.is_pressed('x'):
                    if not was_pressed:
                        sentence.append(actions[1])
                        was_pressed = True
                else:
                    was_pressed = False
                # 3 viz Logic
            #    if res[np.argmax(res)] > threshold:
            #        if len(sentence) > 0:
            #            if actions[np.argmax(res)] != sentence[-1]:
            #                sentence.append(actions[np.argmax(res)])
            #        else:
            #            sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

                # viz probability
                #img = prob_viz(res, actions, img, colors)

            cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(img, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('GestureTrackingTest', img)

            # exit code
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
