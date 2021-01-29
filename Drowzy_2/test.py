#!/usr/bin/env python
import numpy as np
import sys
import cv2
import os
##import pytesseract
##from gtts import gTTS
from PIL import Image
import time 

cap = cv2.VideoCapture(0)
sample=0;
error=0
#time.sleep(2)
while(cap.isOpened()):
#while(True):
    ret, img = cap.read()
    if ret:
        error=1
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',img)
        cv2.imwrite('frame.png',img)
        cv2.waitKey(1)
        sample=sample+1
    if (sample == 2):
         sample =0;
         break

cap.release()
if error ==0:
    print('Camera is interrupted\nPlease execute the script again')
    cv2.destroyAllWindows()
if error ==1:
   print('image is caputured')
##im = Image.open("frame.png")
##text = pytesseract.image_to_string(im,lang = 'eng')
###temp = text.encode('utf-8')
###type(text)
################espeak#####################
##lag = 'en'
##myobj = gTTS(text=text,lang=lag, slow =False)
##myobj.save("test.mp3")
##os.system("mpg321 test.mp3")
#############################################
##print(text)
##cv2.destroyAllWindows()
