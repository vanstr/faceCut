# USAGE
# python show_face.py --detector face_detection_model
# import the necessary packages
import argparse
import collections
import os
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS

# construct the argument parser and parse the arguments
face_mask_path = 'loreta/overlay.jpg'
backround_video_path = "loreta/video.mp4"
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

cameraResWidth = 1280
cameraResHeight = 720
# cameraResWidth = 1920
# cameraResHeight = 1080
# cameraResWidth = 3840
# cameraResHeight = 2160
cameraFPS = 30

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading background video stream")
backgr = cv2.VideoCapture(backround_video_path)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
if not vs.isOpened():
    raise Exception("Could not open video device")

fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
vs.set(cv2.CAP_PROP_FOURCC, fourcc)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, cameraResWidth)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, cameraResHeight)

# Test on video fiel
# vs = cv2.VideoCapture("loreta/example.mp4")
time.sleep(2.0)
screenHeight = 720
screenWidth = 1280

# start the FPS throughput estimator
fps = FPS().start()

overlay = cv2.imread(face_mask_path)
overlay = cv2.resize(overlay, (screenHeight, screenWidth))

blank_image = np.zeros((screenHeight, screenWidth, 3), np.uint8)

# y,y,x,x
last_detected_face_rect = (0, cameraResHeight, 0, cameraResWidth)

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
            # reset_last_detected_coordinates()
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
    # h = float(w / 9 * 16)
    # startY = int(latest_nose_coordinates_y - float(h / 2))
    # endY = int(latest_nose_coordinates_y + float(h / 2))
    return face_searching_frame[startY:endY, startX:endX]


def update_found_face_stat(detections):
    global face_recognition_stat
    print("[DEBUG] is_found_face amount:" + str(len(detections)))
    if len(detections) > 0:
        face_recognition_stat.appendleft(True)
        return True
    else:
        face_recognition_stat.appendleft(False)
        return False


def show_face_of_frame(face_searching_frame):
    global backgr
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
                # updateLastDetectedCoordinates(rect)
                display_face_frame(face)

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
        brate, bframe = backgr.read()
        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if brate == True:
            cv2.imshow("Frame", bframe)
        else:
            print("[INFO] restart background video stream")
            backgr = cv2.VideoCapture(backround_video_path)


def updateLastDetectedCoordinates( rect):
    global last_detected_face_rect
    (startX, startY, endX, endY) = rect
    (startFrameY, endFrameY, startFrameX, endFrameX) = last_detected_face_rect
    last_detected_face_rect = (startY + startFrameY, endY + startFrameY, startX + startFrameX, endX + startFrameX)
    print("[INFO] Updated coordinates to "
          + str(startY + startFrameY) + ":" + str(endY + startFrameY) + " " + str(startX + startFrameX) + ":" + str(
                endX + startFrameX))


def reset_last_detected_coordinates():
    global last_detected_face_rect
    print("[INFO] Reset detected coordinates ")
    last_detected_face_rect = (0, cameraResHeight, 0, cameraResWidth)


def get_face_tracked_area(frame):
    global last_detected_face_rect
    (startY, endY, startX, endX) = last_detected_face_rect
    if startY == 0 and startX == 0 and endY == cameraResHeight:  # TODO improve
        return frame
    else:
        errorY = (endY - startY) / 2
        errorX = (endX - startX) / 2
        erStartX = startX - errorX
        erStartY = startY - errorY
        erEndY = endY + errorY
        erEndX = endX + errorX

        if erStartX < 0:
            erStartX = 0
        if erEndX > cameraResWidth:
            erEndX = cameraResWidth

        if erStartY < 0:
            erStartY = 0
        if erEndY > cameraResHeight:
            erEndY = cameraResHeight

        return frame[erStartY:erEndY, erStartX:erEndX]


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    rate, frame = vs.read()

    focused_frame = frame #get_face_tracked_area(frame)
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
backgr.release()
