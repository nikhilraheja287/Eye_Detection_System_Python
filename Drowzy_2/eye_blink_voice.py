#import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

import os 
import pygame

##import RPi.GPIO as GPIO
##
##GPIO.setwarnings(False)
##GPIO.setmode(GPIO.BCM)
##
##RELAY1 = 14
##RELAY2 = 15
##RELAY3 = 19
##RELAY4 = 26

##GPIO.setup(RELAY1,GPIO.OUT)
##GPIO.setup(RELAY2,GPIO.OUT)
##GPIO.setup(RELAY3,GPIO.OUT)
##GPIO.setup(RELAY4,GPIO.OUT)


##GPIO.output(RELAY1,False)
##GPIO.output(RELAY2,False)
##GPIO.output(RELAY3,False)
##GPIO.output(RELAY4,False)

flag1=0
flag2=0
flag3=0
flag4=0
flag5=0
flag6=0
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
EYE_AR_CONSEC_FRAMES9 =10

EYE_AR_CONSEC_FRAMES15 = 17
EYE_AR_CONSEC_FRAMES20 =25

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
sec3=0
sec9=0
sec15=0
sec20=0
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
 
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame,width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
 
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
			# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
 
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
				sec3=1
##				print(COUNTER)

##			elif COUNTER >= EYE_AR_CONSEC_FRAMES9:
##				TOTAL += 1
##				sec9=1
##				print('Counter {}',format(COUNTER))
##
##			elif COUNTER >= EYE_AR_CONSEC_FRAMES15:
##				TOTAL += 1
##				sec15=1
##				print('Counter {}',format(COUNTER))
##
##
##			elif COUNTER >= EYE_AR_CONSEC_FRAMES20:
##				TOTAL += 1
##				sec20=1
##				print('Counter {}',format(COUNTER))
			# reset the eye frame counter
			COUNTER = 0
			# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	if TOTAL == 1 and flag1==0  :
                sec3=0
                COUNTER=0
                print('What Is Your Name')
                flag1=1
                file = 'what.mp3'
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                
               
##                GPIO.output(RELAY1,False)

	elif TOTAL == 2 and flag2==0 :
                sec9=0
                COUNTER=0
                flag2=1
                print('Help')

                file = 'Help.mp3'
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                
              



	elif TOTAL == 3 and flag3==0 :
                sec15=0
                flag3=1
                print('Medicine Required')

                file = 'medicinerequired.mp3'
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                
##                
                

	elif TOTAL == 4 and flag4==0 :
                flag4=1
                print('Alert')
                file = 'al.mp3'
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                
             
        elif TOTAL == 5 and flag5==0 :
                flag5=1
                print('Good Morning')
                file = 'goodmorning.mp3'
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()

        elif TOTAL == 6 and flag6==0 :
                flag6=1
                print('Welcome')
                file = 'welcome.mp3'
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()


	elif TOTAL == 7 :
                print('Clear')
                flag7=0
                file = 'c.mp3'
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                
                flag1=0
                flag2=0
                flag3=0
                flag4=0
                flag5=0
                flag6=0
                TOTAL=0
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
