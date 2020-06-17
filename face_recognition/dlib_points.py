import cv2
import dlib
import numpy as np
from skimage import io


def detect_crop_and_show(facepath):
    img = io.imread(facepath)
    faces = detector(img, 3)
    print("Number of faces detected: {}".format(len(faces)))

    print("rectangle area:", faces)
    height_max = 0
    width_sum = 0

    for face in faces:
        shape = predictor(img, face)
        print(shape.parts())
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        print("face_landmark:")
        print("lankmark's shape", landmark.shape)
        for idx, point in enumerate(landmark):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(
                img,
                str(idx),
                pos,
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.3,
                color=(
                    0,
                    255,
                    0))

        # 计算矩形框大小
        height = face.bottom() - face.top()
        width = face.right() - face.left()

        # 处理宽度
        width_sum += width

        # 处理高度
        if height > height_max:
            height_max = height
        else:
            height_max = height_max

        # 绘制用来显示人脸的图像的大小
        print(
            "窗口大小 / The size of window:",
            '\n',
            "高度 / height:",
            height_max,
            '\n',
            "宽度 / width: ",
            width_sum)

        img_blank = np.zeros((height_max, width_sum, 3), np.uint8)
        blank_start = 0
        for face in faces:

            height = face.bottom() - face.top()
            width = face.right() - face.left()

            # 填充
            for i in range(height):
                for j in range(width):
                    img_blank[i][blank_start + j] = img[face.top() +
                                                        i][face.left() + j]
            # 调整图像
            blank_start += width

        cv2.namedWindow("img_faces")  # , 2)
        cv2.imshow("img_faces", img_blank)
        cv2.imwrite(r'../testimgs/croped1.jpg', img_blank)
        cv2.waitKey(0)


if __name__ == '__main__':
    predictor_path = r"E:\DESKTOP\shape_predictor_68_face_landmarks.dat"
    faces_path = r"../testimgs/affined1.jpg"

    '''加载人脸检测器、加载官方提供的模型构建特征提取器'''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    detect_crop_and_show(faces_path)

