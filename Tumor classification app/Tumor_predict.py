import PyQt5.QtWidgets
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
import sys
import pydicom
from PIL import Image
import numpy as np
import os

from keras.models import Sequential
from keras.layers import  Dense, Dropout,GlobalAveragePooling2D
import keras
import tensorflow
from keras import applications
import cv2 as cv
from keras_preprocessing.image import img_to_array


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        # Call the inherited classes __init__ method
        super(Ui, self).__init__()
        uic.loadUi('GUI.ui', self)  # Load the .ui file
        # "connect" the button to the variables
        self.button_load_image = self.findChild(QtWidgets.QPushButton, 'ButtonLoadImage')
        self.button_load_image.clicked.connect(self.load_image)

        self.button_predict = self.findChild(QtWidgets.QPushButton, 'ButtonPredict')
        self.button_predict.clicked.connect(self.predict)

        self.label_title = self.findChild(QtWidgets.QLabel, 'LabelTitle')
        self.label_answer = self.findChild(QtWidgets.QLabel, 'LabelAnswer')
        self.label_back = self.findChild(QtWidgets.QLabel, 'label_back')

        self.label_mri = self.findChild(QtWidgets.QLabel, 'label_mri')
        # decide size of the window
        self.setFixedWidth(977)

        self.setFixedHeight(892)
        self.path = ""
        self.img = ""
    #load image and present it on the gui window
    def load_image(self):
        path = ''
        self.label_answer.setText("")
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open File",
                                                      r"C:\Users\danie\OneDrive\שולחן העבודה\Brain tumor",
                                                      "All Files(*)")
        if not fname[0]:
            return

        im = pydicom.dcmread(fname[0])#read the dcm file
        self.img = fname[0]
        im = im.pixel_array.astype(float)
        if np.max(im) != 0:#make pixel values between 0-255
            im = (np.maximum(im, 0) / im.max()) * 255

        final_image = np.uint8(im)
        final_image = Image.fromarray(final_image)
        path_list = fname[0].split("/")
        for i in range(len(path_list) - 1):#create a png format image
            path += path_list[i] + "/"
        path = path + "temp.png"
        final_image.save(path)
        self.path = path
        self.pixmap = QPixmap(path)
        self.label_mri.setPixmap(self.pixmap)
        PyQt5.QtWidgets.QApplication.processEvents()#load the image in the gui window
        if os.path.exists(path):
            os.remove(path)

    def predict(self):
        if not self.img:#check if image is loaded
            self.label_answer.setText("Load MRI scan")
            self.label_answer.setStyleSheet("color: rgb(0, 0, 0);")
            return

        im = load_dicom(self.img)
        y_pred = inception.predict(im)#predict the image

        if y_pred[0][0] > y_pred[0][1]:#print the prediction
            self.label_answer.setText(f"The probability of no tumor is: {y_pred[0][0]:.2f}")
            self.label_answer.setStyleSheet("color: rgb(0, 170, 0);")
        else:
            self.label_answer.setText(f"The probability of a tumor is: {y_pred[0][1]:.2f}")
            self.label_answer.setStyleSheet("color: rgb(250, 0, 0);")


"""Create the Inception V3 model with random weights initialization, without the top and with input shape width = 128 height = 128 and channels = 1, output shape is 2. Add layers to the model, compile. Return the compiled model."""
def create_inception_v3():
    input_shape = (128, 128, 1)
    nclass = 2

    base_model = keras.applications.InceptionV3(weights=None, include_top=False, input_shape=input_shape)
    add_model = Sequential()
    add_model.add(base_model)
    add_model.add(GlobalAveragePooling2D())
    add_model.add(Dropout(0.5))
    add_model.add(Dense(nclass, activation='softmax'))

    model = add_model
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

"""Load .dcm format image and change the value pixels betwwen 0- 255.Resize the image. Retrun the proccesed image."""
def load_dicom(path):
    list_im = []
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)  # make pixel values between 0-255
    img = cv.resize(data, (128, 128))  # resize image
    image = img_to_array(img)
    list_im += [image]
    return np.array(list_im)


# main

inception = create_inception_v3()#Create the model
inception.load_weights(r"C:\Users\danie\PycharmProjects\BTP\myModel")#load model weights
# Create an instance of QtWidgets.QApplication
app = QtWidgets.QApplication(sys.argv)
widget = QtWidgets.QStackedWidget()
window = Ui()  # Create an instance of our class
widget.addWidget(window)
widget.show()
app.exec_()  # Start the application
"delete file"
exit()
