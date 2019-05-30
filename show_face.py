# USAGE
# python show_face.py --detector face_detection_model
# import the necessary packages
import argparse
import collections
import ctypes
import os
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-f", "--fullscreen", type=bool, default=True,
                help="Enter presentation mode in fullscreen")
ap.add_argument("-m", "--minimgwidth", type=int, default=80,
                help="Minimal detected face width size")
args = vars(ap.parse_args())

min_amount_of_detection_in_quee_to_show_face = 30
face_statistic_quee = 40
faceSearchAreaWidth = 600
cameraResWidth = 1920
cameraResHeight = 1080
# cameraResWidth = 3840
# cameraResHeight = 2160
cameraFPS = 30

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
# Check success
if not vs.isOpened():
    raise Exception("Could not open video device")


fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
vs.set(cv2.CAP_PROP_FOURCC, fourcc)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, cameraResWidth)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, cameraResHeight)

time.sleep(2.0)
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
screenHeight = user32.GetSystemMetrics(0)
screenWidth = user32.GetSystemMetrics(1)

# start the FPS throughput estimator
fps = FPS().start()

overlay = cv2.imread('loreta/overlay.jpg')
overlay = cv2.resize(overlay, (screenHeight, screenWidth))

background = cv2.imread('loreta/background.png')
background = cv2.resize(background, (screenHeight, screenWidth))

blank_image = np.zeros((screenHeight, screenWidth, 3), np.uint8)

centerX = cameraResWidth // 2
centerY = cameraResHeight // 2

latest_nose_coordinates_x = centerX
latest_nose_coordinates_y = centerY

is_shown_face_model = False
lastValidFaceRecognition = 0
face_recognition_stat = collections.deque([], face_statistic_quee)
face_recognition_stat.appendleft(False)

print("[INFO] Configurations:")
print("[INFO] - min face detecction = "
      + str(min_amount_of_detection_in_quee_to_show_face)
      + ", " + str(face_recognition_stat))


#
# ser = serial.Serial('COM4')
#
# if ser.isOpen():
#     ser.close()
# ser.open()
# ser.isOpen()


def move_face_model(move_forward):
    # global ser
    print("[INFO] move face:" + str(move_forward))
    # if move_forward:
    #     ser.write("F\n")
    # else:
    #     ser.write("B\n")


def update_face_model_state():
    global face_recognition_stat
    global is_shown_face_model
    successful_recognition_amount = list(face_recognition_stat).count(True)

    print("[DEBUG] successfully recognized faces in queue:" + str(successful_recognition_amount))
    if not is_shown_face_model:
        if successful_recognition_amount > min_amount_of_detection_in_quee_to_show_face:
            show_face_model()
    else:
        if True not in face_recognition_stat:
            reset_nose_coordinates()
            hide_face_model()


def show_face_model():
    global is_shown_face_model
    is_shown_face_model = True
    move_face_model(True)
    print("[DEBUG] Move on face model...")


def hide_face_model():
    global is_shown_face_model
    is_shown_face_model = False
    move_face_model(False)
    print("[DEBUG] Hide face model...")


def display_face_frame(face):
    if args["fullscreen"]:
        face = cv2.resize(face, (screenHeight, screenWidth))
        face = cv2.addWeighted(face, 1, overlay, -1, 0)
        rotated_face = imutils.rotate_bound(face, 270)
        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Frame", rotated_face)
    else:
        cv2.imshow("Debug frame", face)


def get_biggest_face_coordinates(detections, w, h):
    biggest_width = args["minimgwidth"]
    biggest_rect = None

    for i in range(0, detections.shape[2]):
        # extract the confidence associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detection_width = endX - startX
            detection_height = endY - startY
            if biggest_width < detection_width and detection_height > detection_width / 2:
                biggest_width = detection_width
                biggest_rect = (startX, startY, endX, endY)
    return biggest_rect


def get_adapted_face(rect, face_searching_frame):
    (startX, startY, endX, endY) = rect
    # w = endX - startX
    # h = w / 9 * 16
    # startY = int(nose_coordinates[1] - float(h / 2))
    # endY = int(nose_coordinates[1] + float(h / 2))

    return face_searching_frame[startY:endY, startX:endX]


def updateFoundFaceStat(detections):
    global face_recognition_stat
    print("[DEBUG] is_found_face amount:" + str(len(detections)))
    if len(detections) > 0:
        face_recognition_stat.appendleft(True)
        return True
    else:
        face_recognition_stat.appendleft(False)
        return False


def show_face_of_frame(face_searching_frame):
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        face_searching_frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    (h, w) = face_searching_frame.shape[:2]
    # print("[INFO] frame size " + str(w) + " : " + str(h))

    rect = get_biggest_face_coordinates(detections, w, h)
    if rect is not None:
        # extract the face ROI
        face = get_adapted_face(rect, face_searching_frame)
        (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW > 0 and fH > 0:
            face_recognition_stat.appendleft(True)
            if is_shown_face_model:
                display_face_frame(face)
                # updateNoseCoordinates(rect)

        else:
            face_recognition_stat.appendleft(False)
            print("[DEBUG] face to small "
                  + str(fW) + ":" + str(fH)
                  + str(rect)
                  + " expected at least " + str(args["minimgwidth"])
                  )
    else:
        face_recognition_stat.appendleft(False)

    update_face_model_state()

    if not is_shown_face_model and args["fullscreen"]:
        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Frame", blank_image)


def updateNoseCoordinates(rect):
    global latest_nose_coordinates_x
    global latest_nose_coordinates_y
    (startX, startY, endX, endY) = rect
    if startY > 0 and startX > 0 and endY - startY <= faceSearchAreaWidth and endX - startX <= faceSearchAreaWidth:
        nose_coordinates = (float(startX + endX) / 2, float(startY + endY) / 2)
        latest_nose_coordinates_x = latest_nose_coordinates_x - faceSearchAreaWidth / 2 + int(
            nose_coordinates[0])
        latest_nose_coordinates_y = latest_nose_coordinates_y - faceSearchAreaWidth / 2 + int(
            nose_coordinates[1])
        print("[INFO] Updated nose coordinates to "
              + str(latest_nose_coordinates_x) + ":" + str(latest_nose_coordinates_y))


def reset_nose_coordinates():
    global latest_nose_coordinates_x
    global latest_nose_coordinates_y
    print("[INFO] Reset nose coordinates to center")

    latest_nose_coordinates_x = centerX
    latest_nose_coordinates_y = centerY


def get_face_tracked_area(frame):
    startX = latest_nose_coordinates_x - faceSearchAreaWidth / 2
    startY = latest_nose_coordinates_y - faceSearchAreaWidth / 2

    if startX < 0:
        startX = 0
    endX = startX + faceSearchAreaWidth
    if endX > screenWidth:
        endX = screenWidth

    if startY < 0:
        startY = 0
    endY = startY + faceSearchAreaWidth
    if endY > screenHeight:
        endY = screenHeight

    return frame[startY:endY, startX:endX]


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    rate, frame = vs.read()

    focused_frame = frame  # get_face_tracked_area(frame)
    cv2.imshow("focused_frame", focused_frame)
    show_face_of_frame(focused_frame)

    # update the FPS counter
    fps.update()

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
