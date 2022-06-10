from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import cvlib as cv
from detecto import core 
from detecto.visualize import show_labeled_image, detect_video          
# load model
model = core.Model.load('heo_model_weights.pth', ['heo'])

# open webcam
webcam = cv2.VideoCapture(0)

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # apply pig detection
    labels, boxes, scores = model.predict(frame)
   
    filtr_ind=np.where(scores>0.6)
    filtr_scr=scores[filtr_ind]
    filtr_boxes=boxes[filtr_ind]
    num_list = list(filtr_ind[0])
    filtr_labels = []
    for i in num_list:
        filtr_labels.append(labels[i] + "#" + str(i+1))
    show_labeled_image(frame, filtr_boxes, filtr_labels)
    
    # display output
    cv2.imshow("pig detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()