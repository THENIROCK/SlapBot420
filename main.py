# https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python

from itertools import count
from typing import Counter
import cv2
import sys
import random
from PIL import Image
from cv2 import VideoCapture
from cv2 import imshow
import time

cam = VideoCapture(0)

result, image = cam.read()

if result:
    imshow("Target Practice", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)  # returns Rect(x,y,w,h) for each face detected --> these are the pixel locations for the rectangle.

print("[INFO] Found {0} Faces.".format(len(faces)))

counter = 0

for (x, y, w, h) in faces:  # iterate through list of pixel locations
    counter = counter+1
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # get all the pixels within face recognition rectangle
    roi_color = image[y:y + h, x:x + w]
    print("[INFO] Object found. Saving locally.")
    cv2.imwrite(str(counter) + '_faces.jpg', roi_color)


status = cv2.imwrite('faces_detected.jpg', image)
print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

chosen_person = Image.open(str(random.randint(1, counter)) + "_faces.jpg")

chosen_person.show()
