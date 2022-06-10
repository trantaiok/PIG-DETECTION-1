from pyexpat import model
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import time
from keras.models import load_model
import matplotlib.pyplot as plt
from detecto import core, utils, visualize
import numpy as np
import seaborn as sns
from torchvision import transforms as tf
from detecto.visualize import show_labeled_image, plot_prediction_grid, detect_video
from detecto.core import Model
from tensorflow.keras.preprocessing.image import img_to_array

model = core.Model.load('heo_model_weights.pth', ['heo'])

image = utils.read_image('D:/NHAN_DIEN_HEO/NHANDIENHEO/image/heo802.jpg') 
labels, boxes, scores = model.predict(image)
filtr_ind=np.where(scores>0.6)
filtr_scr=scores[filtr_ind]
filtr_boxes=boxes[filtr_ind]
num_list = list(filtr_ind[0])
filtr_labels = []
for i in num_list:
  filtr_labels.append(labels[i] + "#" + str(i+1))
show_labeled_image(image, filtr_boxes, filtr_labels)
print(labels) 
print(boxes)
print(scores)

visualize.detect_live(model) 






