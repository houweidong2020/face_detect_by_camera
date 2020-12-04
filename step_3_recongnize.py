#
# 人脸检测

""" 
注：1. names中存储已经识别人的名字，若该人id为0则他的名字在第一位，id位1则排在第二位，以此类推，这个需要手工设定
注：2. 最终效果为一个绿框，框住人脸，左上角为红色的人名，左下角为红色的概率。 
"""

import cv2


__cascadePath__     = r'D:\works\python\opencv-450\data\haarcascades\haarcascade_frontalface_default.xml'
__training_result__ = r'face_trainer\trainer.yml'
__confidence_hold__ = 100       # 可信度门限，越小容差越小
font                = cv2.FONT_HERSHEY_SIMPLEX


result      = __training_result__
cascadePath = __cascadePath__

faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer  = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(result)

idnum = 0

names = ['HouWeidong', 'HouYitong', "xxx"]

cam  = cv2.VideoCapture(0, cv2.CAP_DSHOW)
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < __confidence_hold__:
            idnum = names[idnum]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            idnum = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

        cv2.putText(img, str(idnum), (x+5, y-5), font, 1, (0, 255, 0), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0, 0, 255), 2)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()