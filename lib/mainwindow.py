import shutil
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from .model import KNN
from .trainset import load_trainset
from face_recognition import *

predictor_model = r'E:/DESKTOP/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)


class Ui_MainWindow(object):

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.all_small_imgs = []
        self.small_img_index = 0
        self.current_img = None
        self.model = KNN()
        self.predicted_label = None
        self.trainset, self.trainlabels = None, None
        self.k = 1
        self.dis_method = 'e'
        self.max_images_per_person = 10

    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1020, 1000)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.centralwidget.setFont(font)
        self.centralwidget.setStyleSheet("font: 12pt \"Consolas\";")
        self.centralwidget.setObjectName("centralwidget")
        self.btn_load_camera = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load_camera.setEnabled(True)
        self.btn_load_camera.setGeometry(QtCore.QRect(210, 550, 161, 51))
        self.btn_load_camera.clicked.connect(self.buttonclicked1)

        self.btn_load_model = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load_model.setEnabled(True)
        self.btn_load_model.setGeometry(QtCore.QRect(210, 660, 161, 51))
        self.btn_load_model.clicked.connect(self.buttonclicked3)

        self.btn_start_train = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start_train.setEnabled(True)
        self.btn_start_train.setGeometry(QtCore.QRect(410, 660, 161, 51))
        self.btn_start_train.clicked.connect(self.buttonclicked4)

        self.btn_save_image = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save_image.setEnabled(True)
        self.btn_save_image.setGeometry(QtCore.QRect(210, 750, 161, 51))
        self.btn_save_image.clicked.connect(self.buttonclicked5)

        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(820, 560, 120, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.choose_k)

        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(630, 560, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setWordWrap(False)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(590, 620, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setWordWrap(False)
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(600, 690, 221, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setWordWrap(False)
        self.label_4.setObjectName("label_4")
        self.s_img1 = QtWidgets.QLabel(self.centralwidget)
        self.s_img1.setGeometry(QtCore.QRect(20, 10, 168, 192))
        self.s_img1.setText("not load")
        self.s_img1.setObjectName("s_img1")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(800, 690, 211, 81))
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        font = QtGui.QFont()
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        item.setFont(font)

        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)

        btn1 = QtWidgets.QPushButton("Euclidean")
        btn1.clicked.connect(self.choose_dis_method)
        item.setBackground(brush)
        self.listWidget.addItem(item)
        self.listWidget.setItemWidget(item, btn1)
        btn2 = QtWidgets.QPushButton("Manhattan")
        btn2.clicked.connect(self.choose_dis_method)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.listWidget.setItemWidget(item, btn2)
        btn3 = QtWidgets.QPushButton("cosine")
        btn3.clicked.connect(self.choose_dis_method)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.listWidget.setItemWidget(item, btn3)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(750, 20, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setWordWrap(False)
        self.label_5.setObjectName("label_5")
        # display the final label of current image
        self.result = QtWidgets.QLineEdit(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(750, 80, 221, 31))
        self.result.setObjectName("result")

        # choose the num of images per person
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(820, 620, 120, 31))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.textChanged.connect(self.choose_max_image_per_person)

        self.s_img2 = QtWidgets.QLabel(self.centralwidget)
        self.s_img2.setGeometry(QtCore.QRect(20, 210, 168, 192))
        self.s_img2.setText("not load ")
        self.s_img2.setObjectName("s_img2")
        self.s_img3 = QtWidgets.QLabel(self.centralwidget)
        self.s_img3.setGeometry(QtCore.QRect(20, 410, 168, 192))
        self.s_img3.setText("not load")
        self.s_img3.setObjectName("s_img3")
        self.s_img4 = QtWidgets.QLabel(self.centralwidget)
        self.s_img4.setGeometry(QtCore.QRect(20, 610, 168, 192))
        self.s_img4.setText("not load ")
        self.s_img4.setObjectName("s_img4")

        self.all_small_imgs = [
            self.s_img1,
            self.s_img2,
            self.s_img3,
            self.s_img4]

        self.b_img = QtWidgets.QLabel(self.centralwidget)
        self.b_img.setGeometry(QtCore.QRect(210, 10, 518, 518))
        self.b_img.setText("")
        self.b_img.setObjectName("b_img")

        self.btn_load_picture = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load_picture.setEnabled(True)
        self.btn_load_picture.setGeometry(QtCore.QRect(410, 550, 161, 51))
        self.btn_load_picture.clicked.connect(self.buttonclicked2)
        self.btn_load_picture.setFont(font)
        self.btn_load_picture.setObjectName("btn_load_picture")
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(750, 140, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setWordWrap(False)
        self.label_10.setObjectName("label_10")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(750, 200, 220, 330))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 940, 26))
        self.menubar.setObjectName("menubar")
        self.menufirst_blood = QtWidgets.QMenu(self.menubar)
        self.menufirst_blood.setObjectName("menufirst_blood")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionfile = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.actionfile.setFont(font)
        self.actionfile.setObjectName("actionfile")
        self.actionsettings = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.actionsettings.setFont(font)
        self.actionsettings.setObjectName("actionsettings")
        self.menufirst_blood.addAction(self.actionfile)
        self.menufirst_blood.addAction(self.actionsettings)
        self.menubar.addAction(self.menufirst_blood.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.set_stylesheet()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "change k"))
        self.label_3.setText(_translate("MainWindow", "image per person"))
        self.label_4.setText(_translate("MainWindow", "choose distance"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.label_5.setText(_translate("MainWindow", "result:"))
        self.btn_load_camera.setText(_translate("MainWindow", "Load Camera"))
        self.btn_load_picture.setText(
            _translate("MainWindow", "load picture "))
        self.btn_save_image.setText(
            _translate("MainWindow", "save image "))
        self.btn_start_train.setText(
            _translate("MainWindow", "start train "))
        self.btn_load_model.setText(
            _translate("MainWindow", "load model "))
        self.btn_load_picture.setText(
            _translate("MainWindow", "load picture "))
        self.label_10.setText(_translate("MainWindow", "details:"))
        self.menufirst_blood.setTitle(_translate("MainWindow", "File"))
        self.actionfile.setText(_translate("MainWindow", "open image"))

    # 载入cemera
    def buttonclicked1(self):
        self.load_camera()

    def buttonclicked2(self):
        imgName, imgType = QFileDialog.getOpenFileName(
            self.MainWindow, "打开图片", "", "*.jpg;;*.pgm;;*.png;;All Files(*)")
        if imgName != "":
            size1 = QSize(518, 518)
            size2 = QSize(168, 192)
            jpg1 = QtGui.QPixmap(imgName).scaled(size1.width(), size1.height())
            jpg2 = QtGui.QPixmap(imgName).scaled(size2.width(), size2.height())

            print("loading", imgName,end=" ")
            self.current_img = cv2.imread(imgName)
            self.current_img = cv2.resize(
                self.current_img, (168, 192), interpolation=cv2.INTER_AREA)
            self.current_img = cv2.cvtColor(self.current_img, cv2.COLOR_RGB2GRAY)
            self.current_img = self.current_img.reshape(1, -1)
            print("shape:",self.current_img.shape)

            self.b_img.setPixmap(jpg1)
            self.all_small_imgs[self.small_img_index].setPixmap(jpg2)
            self.small_img_index += 1
            if self.small_img_index == 4:
                self.small_img_index = 0
        else:
            print("load image canceled")

    def buttonclicked3(self):
        t1=time.time()
        self.trainset, self.trainlabels = load_trainset(r"CroppedYale",max_images_per_person=self.max_images_per_person)
        self.model.fit(self.trainset, self.trainlabels)
        print("time cose:{}".format(time.time()-t1))

    def buttonclicked4(self):
        try:
            self.result.setText("")
            self.textBrowser.setText("")
            # print(self.current_img.shape)  # debug
            self.current_img=self.current_img.flatten()
            self.predicted_label, details = self.model.predict(
                self.current_img, dis_method=self.dis_method,k=self.k)
            self.result.setText(self.predicted_label)
            text = "rank of the k neighbors:\n"
            for index, item in enumerate(details):
                text += "rank{}:{},num of this person:{}\n".format(index, item[0], item[1])
            self.textBrowser.setText(text)

        except AttributeError:
            self.result.setText("no model load")
            self.textBrowser.setText("details")

    def buttonclicked5(self):
        if self.current_img is None:
            QMessageBox.warning(
                self.MainWindow,
                "",
                "no image load",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)
            return
        file_path = QFileDialog.getSaveFileName(
            self.MainWindow, "save image", "", "*.jpg;;*.png;;All Files(*)")
        print("saveing",file_path)
        try:
            shutil.copyfile("./tmp/tmp.jpg",file_path[0])
        except:
            print("save image canceled")

    def load_camera(self):
        self.cap = cv2.VideoCapture(0)
        while (1):
            ret, frame = self.cap.read()
            cv2.imshow("s to capture,q to quit", frame)
            if cv2.waitKey(2) == ord('q'):
                break
            if cv2.waitKey(2) == ord('s'):
                t1 = time.time()

                rectangles = detector(frame, 0)
                cliped_images = get_clip(
                    frame,
                    dlib_predictor=predictor,
                    dlib_rects=rectangles)
                for cliped_image in cliped_images:
                    cv_img = cv2.resize(cliped_image, (168, 192))
                    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2GRAY)
                    try:
                        cv2.imwrite(r"tmp/tmp.jpg", cv_img)
                    except BaseException:
                        print("failed")
                try:
                    self.set_image(r"tmp/tmp.jpg")
                except BaseException:
                    print("failed again")
                print("time cost:", time.time() - t1)

        self.cap.release()
        cv2.destroyAllWindows()

    def set_image(self, path="save.jpg"):
        try:
            img = QImage(path)  # 创建图片实例
        except BaseException:
            img = QImage("2.jpg")  # 创建图片实例

        size1 = QSize(518, 518)
        size2 = QSize(168, 192)

        # 始终保存当前的图片
        self.current_img = cv2.imread(path)
        print(self.current_img.shape)
        self.current_img = cv2.resize(
            self.current_img, (168, 192))
        try:
            self.current_img = cv2.cvtColor(self.current_img, cv2.COLOR_RGB2GRAY)
        except :
            pass
        self.current_img = self.current_img.reshape(1, -1)

        # 修改图片实例大小并从QImage实例中生成QPixmap实例以备放入QLabel控件中
        bigpix = QPixmap.fromImage(img.scaled(size1, Qt.IgnoreAspectRatio))
        smallpix = QPixmap.fromImage(img.scaled(size2, Qt.IgnoreAspectRatio))
        self.b_img.setPixmap(bigpix)
        self.all_small_imgs[self.small_img_index].setPixmap(smallpix)
        self.small_img_index += 1
        if self.small_img_index == 4:
            self.small_img_index = 0

    def choose_dis_method(self):
        button = self.MainWindow.sender()
        buttonpos = button.mapToGlobal(
            QPoint(0, 0)) - self.listWidget.mapToGlobal(QPoint(0, 0))
        item = self.listWidget.indexAt(buttonpos)

        if item.row() == 2:
            self.dis_method = 'c'
        elif item.row() == 1:
            self.dis_method = 'm'
        else:
            self.dis_method = 'e'
        print("now yo are using distance method:",self.dis_method)
        print(item.row())

    def choose_k(self):
        content = self.lineEdit.text()
        try:
            self.k = int(content)
            print(
                "now you set k to:",
                self.k)
        except ValueError:
            self.k = 1
            print("now you using default setting,k=3")

    def choose_max_image_per_person(self):
        content = self.lineEdit_3.text()
        try:
            self.max_images_per_person = int(content)
            print(
                "now you set max_images_per_person to:",
                self.max_images_per_person)
        except ValueError:
            self.max_images_per_person = 10
            print("now you using default setting,10 images per person:")

    def set_stylesheet(self):
        self.b_img.setStyleSheet(
            """QLabel{background:#EEEEEE;border-radius:5px;border:1px solid #222831;}""")
        self.s_img1.setStyleSheet(
            """QLabel{background:#EEEEEE;border-radius:5px;border:1px solid #222831;}""")
        self.s_img2.setStyleSheet(
            """QLabel{background:#EEEEEE;border-radius:5px;border:1px solid #222831;}""")
        self.s_img3.setStyleSheet(
            """QLabel{background:#EEEEEE;border-radius:5px;border:1px solid #222831;}""")
        self.s_img4.setStyleSheet(
            """QLabel{background:#EEEEEE;border-radius:5px;border:1px solid #222831;}""")
        for btn in [
                self.btn_load_picture,
                self.btn_save_image,
                self.btn_start_train,
                self.btn_load_camera,
                self.btn_load_model]:
            btn.setStyleSheet(
                """QPushButton{background:#e7dfd5;border-radius:5px;border:1px solid #EEEEEE;}""")
