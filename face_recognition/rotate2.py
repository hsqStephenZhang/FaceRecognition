import dlib
import cv2
import time
import math
import numpy as np


def get_landmarks(image):

    predictor_model = r'E:/DESKTOP/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    landmarks = []
    for i in range(len(rects)):
        landmarks.append(np.matrix([[p.x, p.y]
                                    for p in predictor(image, rects[i]).parts()]))
    return landmarks


def single_face_alignment(face, landmarks):
    eye_center = ((landmarks[36, 0] + landmarks[45, 0]) * 1. / 2,  # 计算两眼的中心坐标
                  (landmarks[36, 1] + landmarks[45, 1]) * 1. / 2)
    dx = (landmarks[45, 0] - landmarks[36, 0])  # note: right - right
    dy = (landmarks[45, 1] - landmarks[36, 1])

    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(
        eye_center, angle, scale=1)  # 计算仿射矩阵
    print(RotateMatrix)
    align_face = cv2.warpAffine(
        face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
    return align_face


def detect_crop_and_show(landmark, faces, img):
    print("Number of faces detected: {}".format(len(faces)))

    print("rectangle area:", faces)
    height_max = 0
    width_sum = 0

    for face in faces:
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


if __name__ == "__main__":
    face = cv2.imread(r'../testimgs/a.jpg')
    t1=time.time()
    mylandmarks = get_landmarks(face)
    print(mylandmarks)
    img = single_face_alignment(face, mylandmarks[0])
    # cv2.imwrite(r'../affined1.jpg', img)
    print("time:%.4f"%(time.time()-t1))
    # img=cv2.imread("../affined1.jpg")
    # print(img.shape)

