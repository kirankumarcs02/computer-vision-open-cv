from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cam = cv2.VideoCapture(0)
image = ''
# show webcam stream
while cam.isOpened():
    # read frame from cam
    ret, image = cam.read()
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        for landmark, (x, y) in enumerate(shape):
            print('land',landmark)
            if (landmark < 18):
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            elif (landmark < 27):
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
            elif (landmark < 36):
                cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
            elif (landmark < 49):
                cv2.circle(image, (x, y), 1, (255, 255, 0), -1)
            else:
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Output", image)
    cv2.waitKey(1)
