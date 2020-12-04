#
# 人脸数据采集

"""
注：1.在运行该程序前，请先创建一个facedata文件夹并和你的程序放在一个文件夹下。
注：2.程序运行过程中，会提示你输入id，请从0开始输入，即第一个人的脸的数据id为0，第二个人的脸的数据id为1，运行一次可收集一张人脸的数据。
注：3.程序运行时间可能会比较长，可能会有几分钟，如果嫌长，可通过设置__num__大小，默认 100张。

如果实在等不及，可按esc退出，但可能会导致数据不够模型精度下降 
"""

import cv2
import os

# 捕获的最大人脸数量
__num__         = 100
__cascadefile__ = r'D:\works\python\opencv-450\data\haarcascades\haarcascade_frontalface_default.xml'

# 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 级联算法设置
face_detector = cv2.CascadeClassifier(__cascadefile__)

face_id = input('\n>>> Enter user id:')
print('>>> Initializing face capture. Look at the camera and wait ...')

# 创建文件保存路径，这里不做已存在的判断
os.chdir('facedata/')
os.mkdir("user." + str(face_id))
os.chdir("user." + str(face_id))

count = 0
print('>>> ', end='', flush=True)
while True:

    # 从摄像头读取图片
    sucess, img = cap.read()

    # 转为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # 将识别的人脸标识出来，绿色框
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0), 2)

        # 显示人脸采集进度
        count += 1
        if count % 10 == 0:        # 每10张，显示下进度
            print('.', end='', flush=True)

        # 保存图像
        #cv2.imwrite("facedata/user_" + str(face_id) + '/' + str(count) + '.jpg', gray[y: y + h, x: x + w])
        cv2.imwrite(str(count) + '.jpg', gray[y: y + h, x: x + w])

        # 显示人脸捕捉视图
        cv2.imshow('image', img)

    # 保持画面的持续
    k = cv2.waitKey(1)

    if k == 27:   # 通过esc键退出摄像
        break

    elif count >= __num__:  # 得到1000个样本后退出摄像
        print('OK', end='')
        print("\n>>> Captrue image :{}".format(__num__))
        break

# 关闭摄像头
cap.release()
cv2.destroyAllWindows()