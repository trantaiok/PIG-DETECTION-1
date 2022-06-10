from matplotlib import image
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Optimizer
from tensorflow.keras.layers import BatchNormalization
from keras.models import load_model 
import numpy as np
from os import listdir
from numpy import asarray, save
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageDraw, ImageFont
import cv2
from detecto import core, utils, visualize
import matplotlib.image as img


#mention you dataset path
dataset = core.Dataset('D:/NHAN_DIEN_HEO/NHANDIENHEO/image/')
#mention you object label here
model = core.Model(['heo'])



model.fit(dataset, epochs=10, verbose=1)

model.save ('heo_model_weights.pth')

