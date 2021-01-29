#!/usr/bin/env python
import numpy as np
import sys
##import cv2
##import os
from playsound import playsound


from gtts import gTTS
import os

print ("Enter the Text :")
##str=input()
str=raw_input()
print (str)
#while True:
    
#mtext = 'welcome to india welcome to india welcome to india '
lag = 'en'
myobj = gTTS(text=str, lang=lag)
myobj.save("test.mp3")
playsound("test.mp3")
