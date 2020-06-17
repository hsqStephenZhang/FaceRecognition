import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

support_postfix = ["jpg", "jepg", "png", "pgm"]

mainfolder = list(os.walk(r"CroppedYale/"))
persons = mainfolder[0][1]


# 第一步，删除没用的文件
# for i, personname in enumerate(persons):
#     print("loading person", personname)
#     personfolder = list(os.walk(r"CroppedYale" + r"/" + personname))[0][2]
#
#     for item in personfolder:
#         if item.split(".")[-1] not in support_postfix:
#             print(item)
#             os.remove("CroppedYale" + r"/" + personname + r"/" + item)

#  第二步，删除不符合格式要求的图片
# for i, personname in enumerate(persons):
#     print("loading person", personname)
#     personfolder = list(os.walk(r"CroppedYale" + r"/" + personname))[0][2]
#
#     for j, imagetag in enumerate(personfolder):
#         img = cv2.imread(r"CroppedYale" + r"/" + personname + r"/" + imagetag)
#         if img.shape != (192,168, 3):
#             print(img.shape)
#             print("removing", imagetag)
#             os.remove(r"CroppedYale" + r"/" + personname + r"/" + imagetag)
#
