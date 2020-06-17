import cv2
import os
import dlib
import time
import numpy as np


def get_clip(
    img_arr: np.ndarray,
    dlib_predictor: dlib.shape_predictor,
    dlib_rects,
):
    """
    :param img_arr:  the input image, shape of (300,400,3)
    :param dlib_predictor:  predictor from dlib
    :param dlib_rects:  the rectangles detected
    :return:  cliped image
    """
    full_img_arr = dlib.full_object_detections()

    for i in range(len(dlib_rects)):
        full_img_arr.append(dlib_predictor(img_arr, dlib_rects[i]))
    cliped_imgs = dlib.get_face_chips(img_arr, full_img_arr, size=128)
    return cliped_imgs


def get_face(
        img_path,
        output_path,
        dlib_predictor: dlib.shape_predictor,
        dlib_detector: dlib.get_frontal_face_detector):
    original_img = cv2.imread(img_path)
    print(original_img.shape)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # 人脸数rectangles
    rectangles = dlib_detector(original_img, 0)

    cliped_images = get_clip(
        original_img,
        dlib_predictor=dlib_predictor,
        dlib_rects=rectangles)
    for cliped_image in cliped_images:
        # cv_img = cv2.cvtColor(cliped_image, cv2.)
        cv_img = cv2.resize(cliped_image, (168, 192))
        cv2.imwrite(output_path, cv_img)
        print(cv_img.shape)


if __name__ == '__main__':
    t1 = time.time()
    # cv2读取图像
    a="../CropppedYale/myself/save.jpg"
    test_img_path1 = r"E:/DESKTOP/PyProject/FaceRecognition/CroppedYale/myself/save.jpg"
    # test_img_path2 = r"../testimgs/a.jpg"

    predictor_model = r'E:/DESKTOP/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
    predictor = dlib.shape_predictor(predictor_model)

    # basepath = "../train_dir/person3/"
    # storagepath = "../cliped/person3/"
    # for path in list(os.walk(basepath))[0][2]:
    #     try:
    #         get_face(
    #             basepath + path,
    #             storagepath + path,
    #             dlib_detector=detector,
    #             dlib_predictor=predictor)
    #     except RuntimeError:
    #         print("this image couldn't be recognized", path)

    get_face(
        test_img_path1,
        "../test.jpg",
        dlib_detector=detector,
        dlib_predictor=predictor)