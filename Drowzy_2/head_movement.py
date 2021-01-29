import cv2
import numpy as np
import dlib
from math import hypot
from imutils import face_utils


face_landmark_path = 'shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio
right=0
left=0
center=0
blink=0
service=0
count=0
count1=0
s=''
while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) > 0:
                shape = predictor(frame, faces[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

    cv2.imshow("demo", frame)

    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2


##        print(blinking_ratio)
        if blinking_ratio > 5.7:
            blink=blink+1
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))


        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)

##        print('left and right {},{}'.format(gaze_ratio_left_eye,gaze_ratio_right_eye))
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2


        print(gaze_ratio)
        if gaze_ratio < 1:
            right=1
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
##            print('Right : {}'.format(r))
        elif 1 <= gaze_ratio <= 3:
            center=1
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
##            print('Center : {}'.format(c))
        else:
            left=1
            new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
##            print('Left : {}'.format(c))

##        if gaze_ratio <= 0.9:
##            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
##            new_frame[:] = (0, 0, 255)
##        elif 1 < gaze_ratio < 1.7:
##            cv2.putText(frame, "Left", (50, 100), font, 2, (0, 0, 255), 3)
##        else:
##            new_frame[:] = (255, 0, 0)
##            cv2.putText(frame, "center", (50, 100), font, 2, (0, 0, 255), 3)



    if right ==1 and left ==1 and blink==1:
        right =0
        left =0
        blink=0
        print('light')
        data.write('!'.encode())
        count = count + 1

        if count > 1 :
            print('relay off')
            count =0
            data.write('*'.encode())
##        service=1
##        time.sleep(1)
    elif right ==1  and blink==1:
        right =0
        left =0
        blink=0
        data.write('@'.encode())
        print('fan')
        count1=count1+1
        if count1 > 1:
            count1=0
            data.write('&'.encode())
            
##        time.sleep(1)
##        service=2
    elif  left ==1 and blink==1:
        right =0
        left =0
        blink=0
        print('water')
##        time.sleep(1)
##    elif right ==1 and left ==1 and blink==2:
##        right =0
##        left =0
##        blink=0
##        print('Water')
##        file = 'water.mp3'
####        time.sleep(1)
##        pygame.init()
##        pygame.mixer.init()
##        pygame.mixer.music.load(file)
##        pygame.mixer.music.play()
####        playsound('water.mp3')
##        time.sleep(1)
##        service=3

##    elif  1<=blink>=2:
##
####    and right == 0 and left ==0:
##        right =0
##        left =0
####        blink=0
##        print('washroom')
##        blink=0

##        if  2<=blink>=3  and center == 1:
##            print('wwwwwwwww')
####            blink=0
##        elif  5<=blink>=9 and center == 1:
##            right =0
##            left =0
##            blink=0
##            center=0
##            print('washroom')

    elif  3<=blink>=5 :
        print('emergency')
        blink=0
##        message = client.messages.create( 
##                              from_='+18172647952',  
##                              body='Emwrgency',      
##                              to='+918197243697' 
##                          )
##        service=1
    
    elif  1<=blink>=2:

##    and right == 0 and left ==0:
        right =0
        left =0        
 
        
##        time.sleep(1)

##        time.sleep(1)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        right =0
        left =0
        blink=0

    if service == 1 :

        count =count+1
##        print('Emergency')

        while count:
            count =count+1
##            print('Responce not recived')
            playsound('beep.mp3')
            time.sleep(0.1)
            playsound('beep.mp3')
            time.sleep(0.1)
            playsound('beep.mp3')
            time.sleep(0.1)
            service=0
            if count == 3:
                count=0
####                client.api.account.messages.create(
####    to="+918867056017",
####    from_="+2013808315",
####    body="Emergency Hurry up !")
##                if count > 5:
##                    print('Enter the response ')
##                    s=raw_input()
##                    if s=='y':
##                      count=0
##                      print('responded')
##              
##
##    if service == 2 :
##
##        count1 =count1+1
##        print('Enter ')
##
##        while count1:
##            count1 =count1+1
##            print('Responce not recived')
##            playsound('beep.mp3')
##            time.sleep(0.1)
##            playsound('beep.mp3')
##            time.sleep(0.1)
##            playsound('beep.mp3')
##            time.sleep(0.1)
##            service2=0
##            if count1 == 3:
####                account_sid = "AC445e29a098a62bc4084ad155a80213bc"
####                auth_token = "df9894f1181053945c57364c05ae6bb0"
####    
####                client = Client(account_sid, auth_token)
####                client.api.account.messages.create(
####    to="+918867056017",
####    from_="+2013808315",
####    body="Emergency Hurry up !")
##                if count1 > 5:
##                    print('Enter the response ')
##                    s=input()
##                    if s=='y':
##                      count1=0
##                      print('reasponded')
              
####################################################
##    if center ==1  and blink==1:
##        right =0
##        left =0
##        blink=0
##        center
##        print('emergency')
##    if center ==1  and blink==2:
##        right =0
##        left =0
##        blink=0
##        center
##        print('food')
##    if center ==1 and left ==1 and blink==1:
##        right =0
##        left =0
##        blink=0
##        center
##        print('washroom')
##    if center ==1 and left ==1 and blink==2:
##        right =0
##        left =0
##        blink=0
##        center
##        print('tablets')


    cv2.imshow("Frame", frame)
    cv2.imshow("New frame", new_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if cv2.waitKey(1) & 0xFF == ord('r'):
        print('r')
        blink=0
        right=0
        left=0
        count=0
        count1=0
cap.release()
cv2.destroyAllWindows()
