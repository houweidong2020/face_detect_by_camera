从视频文件中进行人脸识别

应用过程：
人脸数据采集 --》 人脸数据训练 --》 人脸检测

1.FaceDetection,人脸检测， 检测摄像头下的人脸，标注人脸和眼睛位置     
2.FaceDataCollect,人脸数据采集  face_datacollect    从摄像头中提取人脸信息，存储在 facedata/user.xxx.jpg
3.face_training,人脸数据训练
4.face_recognition 人脸识别