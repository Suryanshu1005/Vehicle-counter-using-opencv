import enum
import numpy as np
import cv2

#detect web camera
cap = cv2.VideoCapture('video.mp4')

min_width_rect = 80 #min width rect
min_height_rect = 80 #min height

count_line_pos = 550

#Initialise algo (Substractor)

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1

    return cx,cy

detect = []
offset = 6    #allowable error between pixels 
counter = 0


while True:
    ret,frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilateada = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernal)
    dilateada = cv2.morphologyEx(dilateada, cv2.MORPH_CLOSE, kernal)
    counterShape,h = cv2.findContours(dilateada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow('Detector',dilateada)
    cv2.line(frame1,(5,count_line_pos),(1260, count_line_pos),(255,122,0),2)

    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255), 2) 

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,20,(0,255,0), 2)

        for (x,y) in detect:
            if y<(count_line_pos+offset) and y>(count_line_pos-offset):
                counter += 1
                cv2.line(frame1,(5,count_line_pos),(1260, count_line_pos),(0,122,255),2)
                detect.remove((x,y))
                print("Vehicle Counter : "+str(counter))

    cv2.putText(frame1,"Vehicle Counter: "+str(counter),(75,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),4)


    cv2.imshow('Video Original',frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()