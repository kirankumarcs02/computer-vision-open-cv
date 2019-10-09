from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ratio = (A + B) / (2.0 * C)
    print("ratio", ratio)
    return ratio


thresh = 0.25
frame_check = 200
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36,42)


cap = cv2.VideoCapture(0)
flag = 0
is_eye_closed = True
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        left_eye_ratio = eye_aspect_ratio(leftEye)
        right_eye_ratio = eye_aspect_ratio(rightEye)
        avg = (left_eye_ratio + right_eye_ratio) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        for (x_l,y_l) in leftEye:
            cv2.circle(frame,(x_l, y_l), 1, (0, 0, 255), -1)
        for (x_l,y_l) in rightEye:
            cv2.circle(frame,(x_l, y_l), 1, (0, 0, 255), -1)

        if avg < thresh:
            print("-----------EYE CLOSED------------")
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # print ("Drowsy")
        elif avg > thresh:
            print("-----------EYE OPEN------------")
            flag = 0
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()
