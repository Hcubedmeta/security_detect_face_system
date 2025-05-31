import cv2
import urllib.request
import numpy as np
import RPi.GPIO as GPIO
import time

# Thiết lập GPIO
RELAY_PIN = 17  # Chỉnh lại đúng chân GPIO bạn dùng
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  # Ban đầu tắt relay

# Tải các mô hình Haar cascade
f_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

url = 'http://192.168.1.7/cam-hi.jpg'

cv2.namedWindow("Live Transmission", cv2.WINDOW_AUTOSIZE)

face_detected = False
last_detect_time = 0
RELAY_TIMEOUT = 5  # Thời gian giữ relay bật (giây)

try:
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

        faces = f_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            face_detected = True
            last_detect_time = time.time()
            GPIO.output(RELAY_PIN, GPIO.HIGH)  # Bật relay

        # Nếu đã bật relay nhưng quá thời gian, tắt relay
        if face_detected and (time.time() - last_detect_time > RELAY_TIMEOUT):
            GPIO.output(RELAY_PIN, GPIO.LOW)
            face_detected = False

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow("Live Transmission", img)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Đã dừng bởi người dùng")

finally:
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Đảm bảo relay được tắt
    GPIO.cleanup()
    cv2.destroyAllWindows()
