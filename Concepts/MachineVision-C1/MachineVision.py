#imports
import cv2
import mediapipe

#pretrained models
mp_hand = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils

#main loop
#detects and draws the hands
webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    succes, img = webcam.read()

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = mp_hand.Hands().process(img)

    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img,hand,connections=mp_hand.HAND_CONNECTIONS)

    cv2.imshow("Cas",img)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

#release resources
webcam.release()
cv2.destroyAllWindows()