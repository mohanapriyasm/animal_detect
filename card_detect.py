import numpy as np
import cv2
import sys

imagePath = sys.argv[1]

card_cascade = cv2.CascadeClassifier('/home/mohana/Documents/animal_detection/cascade.xml')

img = cv2.imread(imagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cards = card_cascade.detectMultiScale(
     gray, 
     scaleFactor=1.1, 
     minNeighbors=1,
     minSize=(30,30),
     flags = cv2.CASCADE_SCALE_IMAGE
)

for (x,y,w,h) in cards:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0),2)
    
cv2.imshow('Image', img)
cv2.waitKey(0)
