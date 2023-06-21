import cv2
import mediapipe as mp

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

# Load the cascade\
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    # Read the frame
    ret, img = cap.read()
    #img = cv2.resize(img, (600, 400))
    # Convert to grayscale
    results = pose.process(img)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # print(results.pose_landmarks)
    # Display
    text = "Detecting..."
    text2 = "Detecting..."
    lm = results.pose_landmarks

    image_height, image_width, c = img.shape
    if lm != None:
        shoulderHeight = lm.landmark[mp_pose.PoseLandmark(11)].y * image_height

        rHandHeight = lm.landmark[mp_pose.PoseLandmark(16)].y * image_height
        lHandHeight = lm.landmark[mp_pose.PoseLandmark(15)].y * image_height
        rHandWidth = lm.landmark[mp_pose.PoseLandmark(16)].x * image_width
        lHandWidth = lm.landmark[mp_pose.PoseLandmark(15)].x * image_width
        lshoulderWidth = lm.landmark[mp_pose.PoseLandmark(11)].x * image_width
        rshoulderWidth = lm.landmark[mp_pose.PoseLandmark(12)].x * image_width
        #print(lshoulderWidth ,' L : R ',rshoulderWidth)
        shoulderLength = lshoulderWidth - rshoulderWidth
        if rHandHeight < shoulderHeight and lHandHeight > shoulderHeight:
            text = "Rechterhand boven schouder" #, str(shoulderHeight) , " H: " , str(rHandHeight)
        elif lHandHeight <shoulderHeight and rHandHeight > shoulderHeight:
            text = "Linkerhand boven schouder"#, str(shoulderHeight) , " H: " , str(lHandHeight)
        elif lHandHeight <shoulderHeight and rHandHeight < shoulderHeight:
            text = "Beide handen boven schouder" #, str(shoulderHeight) , " lH: " , str(lHandHeight) , " RH: " , str(rHandHeight)
        else:
            text = "Detecting..."
        if shoulderLength/2 > (lHandWidth - rHandWidth)/2:
            text2 = ' HANDEN BIJ ELKAAR'
        else:
            text2 = 'HANDEN VAN ELKAAR'
    # Stop if escape key is pressed
    x, y, w, h = 0, 0, image_width, 50

    # Create background rectangle with color
    cv2.rectangle(img, (x, x), (x + w, y + h), (50, 50, 50), -1)
    coordinates = (50, 35)
    coordinates2 = (750, 35)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 155, 155)
    thickness = 2
    img = cv2.putText(img, 'Locatie: '+text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    img = cv2.putText(img, 'Afstand: '+text2, coordinates2, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# Release the VideoCapture object
cap.release()
