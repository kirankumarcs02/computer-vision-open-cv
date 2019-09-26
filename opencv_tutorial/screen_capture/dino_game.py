import time
import cv2
import mss
import numpy
import numpy as np

screen_capture_once = False

template = cv2.imread('dino_1.PNG',0)
template_w, template_h = template.shape[::-1]
method = eval('cv2.TM_CCOEFF_NORMED')

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 800, "height": 640}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        real_image = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        print('min_loc', min_loc)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        cv2.rectangle(real_image, top_left, bottom_right, (0, 0, 255), 2)

        for index in range(1,7):
            temp_cactus = cv2.imread('cactus_'+ str(index)+'.jpg', 0)
            res = cv2.matchTemplate(img, temp_cactus, cv2.TM_CCOEFF_NORMED)
            w, h = temp_cactus.shape[::-1]
            loc = np.where(res >= 0.8)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(real_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)

        cv2.imshow("screen", real_image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

