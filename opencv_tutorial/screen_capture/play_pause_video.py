import cv2
cap = cv2.VideoCapture('test.mp4')

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xff
    if not ret:
        break
    if key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            cv2.imshow('frame', frame)
            if key2 == ord('p'):
                break
    cv2.imshow('frame',frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
