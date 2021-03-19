import cv2 as cv

cap = cv.VideoCapture(0)

face_casade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_casade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread('face-sample.png')

while True:
    ret, frame = cap.read() # frameの中に画像が入ってます
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_casade.detectMultiScale(gray_image, 1.3, 5)
    print(len(faces))

    for (x, y, w, h) in faces:
        # enclose blue circle around face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        eye_gray = gray_image[y:y+h, x:x+w]
        eye_color = frame[y:y+h, x:x+w]
        eyes = eye_casade.detectMultiScale(eye_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()