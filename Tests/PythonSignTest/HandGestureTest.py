import os

import cv2
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
pathImages = sorted(os.listdir('Img'), key=len)
print(pathImages)
pTime = 0
detector = HandDetector(detectionCon=0.8, maxHands=1)
while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    hands, img = detector.findHands(img, flipType=True)
    text = "none"
    text2 = "none"
    orientation = "none"

    if hands:
        hand = hands[0]
        type = hand['type']
        lmlist = hand['lmList']
        xPinky = lmlist[17][0]
        xThumb = lmlist[2][0]
        fingers = detector.fingersUp(hand)
        print(fingers)
        text2 = str(fingers)
        if xPinky > xThumb:
            if type == 'Right':
                orientation ='BACK'
            else:
                orientation = 'PALM'
        else:
            if type == 'Right':
                orientation = 'PALM'
            else:
                orientation = 'BACK'
        if fingers == [1,1,0,0,1]  and orientation == 'PALM' :
            text = 'I LOVE YOU'
        if fingers == [0, 0, 1, 0, 0] or fingers == [1,0,1,0,0]:
            text ='FUCK TENSOR FLOW'
    #cv2.putText(img,f'FPS: {int(fps)}',(40,70), cv2.FONT_HERSHEY_COMPLEX,
    image_height, image_width, c = img.shape
    x, y, w, h = 0, 0, image_width, 50
    cv2.rectangle(img, (x, x), (x + w, y + h), (50, 50, 50), -1)
    coordinates = (50, 35)
    coordinates2 = (800, 35)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .9
    color = (255, 155, 155)
    thickness = 2
    img = cv2.putText(img, 'ASL: ' + text +' | ORIENTATION: '+ orientation, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    img = cv2.putText(img, ' | FINGERS: ' + text2, coordinates2, font, fontScale, color, thickness, cv2.LINE_AA)
    #            3, (255,0,255),3)
    cv2.imshow("Img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break