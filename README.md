## 使用方法

cd FaceRecognition

pip install -r requirements.txt



## 说明

1.face recognition中，其实只是用了rotate_clip.py文件，其余的是一开始测试的时候写的

2.使用的训练集为CroppedYale,是耶鲁大学的一个人脸识别训练集，包含了不同的光线暗下已经裁剪好的人脸的图像

3.GUI界面是用QT creator设计，配合Pyqt5进行功能设置

4.model采用了knn，有两个主要的参数，分别是k,dis_method(采用的核函数)，可以使用自己设计的模型

5.如果想要动态识别物体的话，建议使用YoloV4或者YoloV3

6.有些文件中的路径可能需要手动修改

