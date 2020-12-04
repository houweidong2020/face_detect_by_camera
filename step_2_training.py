#
# 人脸数据训练

""" 
注：1.运行该程序前，请在人脸识别文件夹下创建face_trainer文件夹。
注：2.训练后的文件会保存在目录中。 
"""

import numpy as np
from PIL import Image
import os
import cv2

# 人脸数据路径
__path__            = 'facedata'
__cascadefile__     = r'D:\works\python\opencv-450\data\haarcascades\haarcascade_frontalface_default.xml'
__training_result__ = r'face_trainer\trainer.yml'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector   = cv2.CascadeClassifier(__cascadefile__)

def getImagesAndLabels(path):
    dirs  = [os.path.join(path, f) for f in os.listdir(path)]  # join函数的作用？
    faceSamples = []
    ids         = []

    # 首先扫描文件夹
    for dir in dirs:
        imagePaths = os.listdir(dir)
        #os.chdir(dir)

        # 扫描指定文件加下的文件
        for imagePath in imagePaths:
            file = dir + '/' + imagePath
            PIL_img   = Image.open(file).convert('L')   # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            #id        = int(os.path.split(imagePath)[-1].split(".")[1])  # <--- 继续，待突破
            id        = int(os.path.split(dir)[-1].split(".")[1]) 
            faces     = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x: x + w])
                ids.append(id)
                
    return faceSamples, ids


path   = __path__
result = __training_result__
print('\n>>>Training faces. It will take a few seconds. Wait ...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

#os.chdir(os.path.abspath(os.path.dirname(os.getcwd())))
recognizer.write(result)
print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))