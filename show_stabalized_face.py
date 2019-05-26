# USAGE
# python show_stabalized_face.py --shape-predictor shape_predictor_68_face_landmarks.dat
# import the necessary packages

import argparse
import collections
import ctypes
import time

import cv2
import dlib
import imutils
import numpy as np
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.video import FPS
from imutils.video import VideoStream

################### User configurable properties  ###################
#user32 = ctypes.windll.user32
wishedFaceImgWidth = 320 #int(user32.GetSystemMetrics(0) / 4)
wishedFaceImgHeight = 240 #int(user32.GetSystemMetrics(1) / 4)
minImgSize = int(wishedFaceImgWidth / 6)  # filter out users located too far
imgWidth = wishedFaceImgWidth
is_full_screen_mode = False
min_amount_of_detection_in_quee_to_show_face = 5
face_statistic_quee = 15
#####################################################################


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())




is_shown_face_model = False
lastValidFaceRecognition = 0
faceRecognitionStat = collections.deque([], face_statistic_quee)
faceRecognitionStat.appendleft(False)

print("[INFO] Configurations:")
print("[INFO] - img resolution 0 = " + str(wishedFaceImgWidth) + " 1 = " + str(wishedFaceImgHeight))
print("[INFO] - min face detecction = 5, " + str(faceRecognitionStat))


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


def is_found_face(found_faces_num, face_recognition_stat):
    print("[DEBUG] is_found_face amount:" + str(found_faces_num))
    if found_faces_num > 0:
        face_recognition_stat.appendleft(True)
        return True
    else:
        face_recognition_stat.appendleft(False)
        return False


def update_face_model_state(face_recognition_stat):
    global is_shown_face_model
    successful_recognition_amount = list(face_recognition_stat).count(True)

    print("[DEBUG] successful_recognition_amount" + str(successful_recognition_amount))
    if not is_shown_face_model:
        if successful_recognition_amount > min_amount_of_detection_in_quee_to_show_face:
            show_face_model()
    else:
        if True not in face_recognition_stat:
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


def get_the_biggest_face(rects):
    biggest_width = minImgSize
    biggest_rect = None
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if biggest_width < w:
            biggest_width = w
            biggest_rect = rect
    return biggest_rect


# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = dlib.get_frontal_face_detector()

# initialize FaceAligner
print("[INFO] initialize FaceAligner")
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredLeftEye=(0.38, 0.38), desiredFaceWidth=imgWidth)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # load the input image, resize it, and convert it to grayscale
    image = imutils.resize(frame, width=imgWidth)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 0)
    detectedFacesNum = len(rects)
    print("[INFO] detected " + str(detectedFacesNum) + " faces")

    if is_found_face(detectedFacesNum, faceRecognitionStat) and is_shown_face_model:
        # find the closest face
        rect = get_the_biggest_face(rects)
        if rect is not None:
            (x, y, w, h) = rect_to_bb(rect)
            new_image = image[y:y + h, x:x + w]
            (hNotNull, wNotNull) = new_image.shape[:2]
            if w > 0 and wNotNull > minImgSize and hNotNull > minImgSize:
                faceOrig = imutils.resize(new_image, width=wishedFaceImgWidth)
                faceAligned = fa.align(image, gray, rect)
                cut = int((wishedFaceImgWidth - wishedFaceImgHeight) / 2)
                (h2, w2) = faceAligned.shape[:2]
                faceAligned = faceAligned[0:wishedFaceImgWidth, cut:(w2 - cut)]
                # display the output images
                if is_full_screen_mode:
                    rotated_face = imutils.rotate_bound(faceAligned, 270)
                    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("Frame", rotated_face)
                else:
                    cv2.imshow("Frame", faceAligned)

    update_face_model_state(faceRecognitionStat)

    if not is_shown_face_model:
        blank_image = np.zeros((wishedFaceImgHeight, wishedFaceImgWidth, 3), np.uint8)
        if is_full_screen_mode:          
            cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Frame", blank_image)
        else:
            cv2.imshow("Frame", blank_image)

    # update the FPS counter
    fps.update()

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
