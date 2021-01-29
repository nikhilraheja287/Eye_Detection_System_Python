#!/bin/sh

Text_To_Display()
{
echo "###############################################################################"
echo " 										   "
echo "			Choose the option for the execution"
echo " 										   "
echo "			1. Blink to Home Automation                            "
echo " 										   "
echo "			2. Blink to Voice						"

echo " 										   "
echo " 									           "
echo "#############################################################################"
}
while :
do
 Text_To_Display
 #echo "Choose option to enter : "
 read INPUT_STRING
 case $INPUT_STRING in 
	1)
		echo "You are executing the Blink to Home Automation "
		python /home/pi/Desktop/blink-detection/eye_blink.py --shape-predictor shape_predictor_68_face_landmarks.dat
		;;
	2)
	 	echo "You are executing the Blink to Voice"
		python /home/pi/Desktop/blink-detection/eye_blink_voice.py --shape-predictor shape_predictor_68_face_landmarks.dat
		;;

	q)
	 	echo "Programm is terminated"
		;;
 esac
done 
echo "complete"
