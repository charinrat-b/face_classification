from mtcnn import MTCNN
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import os
from time import time
import numpy as np
from keras.models import load_model
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from keras import backend as K
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import MaxPool2D
import ssl
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
from PIL import Image
ssl._create_default_https_context = ssl._create_unverified_context

folder_name = os.listdir('/content/face')
persons=[]
detector = MTCNN()

for person_name in folder_name:
    if person_name != '.DS_Store':
      folder_path = os.listdir('/content/face' + '/' + person_name)
      for filename in folder_path:   
        _, dot = os.path.splitext(filename)
        if dot == ".bmp":
          persons.append((person_name,
          str('/content/face' + '/' + person_name) + '/' + filename))
         
person_df = pd.DataFrame(data=persons, columns=['name','img'])
person_count = person_df['name'].value_counts()

im_size = 224
images =[]
labels =[]
required_size = (224,224)
path = '/content/face/'
filenames = []
for i in folder_name:
    if i != '.DS_Store':
        data_path = path + str(i)
        filenames = [i for i in os.listdir(data_path)]
        for f in filenames:
            img = cv2.imread(data_path + '/' + f)
            #img = cv2.resize(img,(im_size,im_size))
            faces = detector.detect_faces(img)
            for box in faces:
              bounding_box = faces[0]['box']
              x1,y1,w,h = faces[0]['box']
              x2, y2 = x1+w, y1 + h
              keypoints = faces[0]['keypoints']
              face_boundary = img[y1:y2, x1:x2]
              
              # resize pixels to the model size
              face_image = Image.fromarray(face_boundary)
              face_image = face_image.resize(required_size)
              face_array = np.asarray(face_image)
              #cv2_imshow(face_array)
              # cv2.rectangle(img,
              #     (bounding_box[0],bounding_box[1]),
              #     (bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3]),
              #     (0,155,255),2)
  
              # cv2.circle(img,(keypoints['left_eye']),2,(0,155,255),2)
              # cv2.circle(img,(keypoints['right_eye']),2,(0,155,255),2)
              # cv2.circle(img,(keypoints['nose']),2,(0,155,255),2)
              # cv2.circle(img,(keypoints['mouth_left']),2,(0,155,255),2)
              # cv2.circle(img,(keypoints['mouth_right']),2,(0,155,255),2)
              #cv2_imshow(img)
            
              images.append(face_array)
              labels.append(i)


images = np.asarray(images)


images = images.astype('float32')/255.0


y = person_df['name'].values
y_labelencod = LabelEncoder()
y = y_labelencod.fit_transform(y)
print(y.shape)

y = y.reshape(-1,1)
onehot = OneHotEncoder(categories='auto')
Y = onehot.fit_transform(y)
print(Y.shape)

images,Y = shuffle(images,Y, random_state=1)

train_x,test_x,train_y,test_y = train_test_split(images,Y, test_size=0.2)

def VGG16():
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(20, activation='softmax', name='output'))
    return model

model = VGG16()
model.summary()

# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# checkpoint = keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")
IMG_SHAPE = (im_size,im_size, 3)
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
#opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])
history = model.fit(train_x,train_y, epochs=50,batch_size=20, validation_split=0.2)






