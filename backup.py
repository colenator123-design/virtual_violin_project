import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import time


cap=cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() 
mpDraw = mp.solutions.drawing_utils

kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()
image = np.zeros((480, 640, 3), np.uint8)
image[:] = (128, 128, 128)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose

state = "release"
holdAngle = 10
signal = -1

pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5)
def FindAngleF(a,b,c):    
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang<0 :
      ang=ang+360
    if ang >= 360- ang:
        ang=360-ang
    return ang


while(True):
    ret,frame=cap.read()
    frame = cv2.resize(frame,(1280,720))
    
    if not ret:
        break
    
    h, w, c = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    rst = hands.process(frameRGB)
    if rst.multi_hand_landmarks:
        
        
        
        imgH,imgW=frame.shape[0],frame.shape[1]
        results = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5).process(frame) #偵測身體
        #左手軸3點->11,13,15
        
        shoulder = np.array([results.pose_landmarks.landmark[12].x*imgW,results.pose_landmarks.landmark[12].y*imgH])
        elbow = np.array([results.pose_landmarks.landmark[14].x*imgW,results.pose_landmarks.landmark[14].y*imgH])
        wrist = np.array([results.pose_landmarks.landmark[16].x*imgW,results.pose_landmarks.landmark[16].y*imgH])
        leftShoulder = np.array([results.pose_landmarks.landmark[11].x*imgW,results.pose_landmarks.landmark[11].y*imgH])
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks, #點
            mp_pose.POSE_CONNECTIONS, #連線
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
          )
        if (results.pose_landmarks.landmark[12] == None or results.pose_landmarks.landmark[14] == None or results.pose_landmarks.landmark[16] == None):
          continue;
        curAngle = FindAngleF(shoulder, elbow, wrist)
        x14,y14=round((1-results.pose_landmarks.landmark[14].x)*imgW),int(results.pose_landmarks.landmark[14].y*imgH)
        if (x14<imgW and x14>0) and (y14<imgH and y14>0):
          cv2.putText(frame, str(round(curAngle,2)) , (x14,y14), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        if (state == "release"):
          if (curAngle > holdAngle + 20):
            state = "bowing"
            if (wrist[1] > elbow[1] and wrist[1] > shoulder[1]):
              signal = 1
              pygame.mixer.init()
              pygame.mixer.music.load('a.mp3')
              pygame.mixer.music.play()
            elif (wrist[0] < shoulder[0] and wrist[0] > elbow[0]):
              if(elbow[1] > shoulder[1]):
                signal = 2
                pygame.mixer.init()
                pygame.mixer.music.load('g.mp3')
                pygame.mixer.music.play()
              else:
                signal = 3
                pygame.mixer.init()
                pygame.mixer.music.load('c.mp3')
                pygame.mixer.music.play()
            elif (wrist[0] > shoulder[0]):
              signal = 4
              pygame.mixer.init()
              pygame.mixer.music.load('c.mp3')
              pygame.mixer.music.play()            
          if (curAngle < holdAngle -5):
            holdAngle = FindAngleF(shoulder, elbow, wrist)
        elif (state == "bowing"):
          if (curAngle < holdAngle - 20):
            state = "release"
            signal = -1
            pygame.mixer.music.stop()
          if (curAngle > holdAngle + 5):
            holdAngle = FindAngleF(shoulder, elbow, wrist)
        cv2.putText(frame,  str(state) , (30,40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(frame,  str(signal) , (30,60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  
        
        
        #for handLms in rst.multi_hand_landmarks:
            #mpDraw.draw_landmarks(frame, handLms)
            # rst.multi_hand_landmarks:
        for hand in rst.multi_hand_landmarks:
                            mpDraw.draw_landmarks(frame, hand)
                            flag=0
                            vstring=0
                            position=0
                            for i in range(7,-1,-1):  
                                for num in range(20,4,-4):
                                    x = hand.landmark[num].x * w
                                    y = hand.landmark[num].y * h              
                                    a=25*i
                                    b=30*i
                                    c=25*(i+1)
                                    d=30*(i+1)
                                    if x < 980-a and y < 470+d and x > 980-c and y > 470+b:
                                        cv2.rectangle(frame, (980-a, 470+b), (980-c, 470+d), (0, 0, 255), 3, cv2.LINE_AA)
                                        flag=1
                                        vstring=1
                                        position=i
                                        break
                                if(flag==1):
                                    break
                            for i in range(7,-1,-1):  
                                for num in range(20,4,-4):
                                    x = hand.landmark[num].x * w
                                    y = hand.landmark[num].y * h              
                                    a=25*i
                                    b=30*i
                                    c=25*(i+1)
                                    d=30*(i+1)
                                    if x < 955-a and y < 435+d and x > 955-c and y > 435+b:
                                        cv2.rectangle(frame, (955-a, 435+b), (955-c, 435+d), (0, 0, 255), 3, cv2.LINE_AA)                   
                                        flag=1
                                        vstring=2
                                        position=i
                                        break
                                if(flag==1):
                                    break  
                            for i in range(7,-1,-1):   
                                for num in range(20,4,-4):
                                    x = hand.landmark[num].x * w
                                    y = hand.landmark[num].y * h             
                                    a=25*i
                                    b=30*i
                                    c=25*(i+1)
                                    d=30*(i+1)
                                    if x < 930-a and y < 400+d and x > 930-c and y > 400+b:
                                        cv2.rectangle(frame, (930-a, 400+b), (930-c, 400+d), (0, 0, 255), 3, cv2.LINE_AA)                     
                                        flag=1
                                        vstring=3
                                        position=i
                                        break
                                if(flag==1):
                                    break
                            for i in range(7,-1,-1):   
                                for num in range(20,4,-4):
                                    x = hand.landmark[num].x * w
                                    y = hand.landmark[num].y * h             
                                    a=25*i
                                    b=30*i
                                    c=25*(i+1)
                                    d=30*(i+1)
                                    if x < 905-a and y < 365+d and x > 905-c and y > 365+b:
                                        cv2.rectangle(frame, (905-a, 365+b), (905-c, 365+d), (0, 0, 255), 3, cv2.LINE_AA)
                                        flag=1
                                        vstring=4
                                        position=i
                                        break
                                if(flag==1):
                                    break
                                
    frame = cv2.flip(frame, 1)
    
    cv2.line(frame,(300,470),(500,710),(0,0,0),3)
    cv2.line(frame,(325,435),(525,675),(0,0,0),3)
    cv2.line(frame,(350,400),(550,640),(0,0,0),3)
    cv2.line(frame,(375,365),(575,605),(0,0,0),3)
    for i in range(1,8):
        a=25*i
        b=30*i
        cv2.circle(frame,(300+a,470+b),2,(255,255,255),4)
        cv2.circle(frame,(325+a,435+b),2,(255,255,255),4)
        cv2.circle(frame,(350+a,400+b),2,(255,255,255),4)
        cv2.circle(frame,(375+a,365+b),2,(255,255,255),4)
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()