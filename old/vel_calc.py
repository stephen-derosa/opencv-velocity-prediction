import numpy as np
import cv2 as cv
import argparse
import os
from time import sleep
import matplotlib.pyplot as plt


#sample comment change
count = 0
cent_X = []
cent_Y = []
vel_set = []
frame_set = []

fps = 60
in_per_pixel = (11+8*12)/(805-455)

max_value = 255

low_H = 0
low_S = 0
low_V = 0
high_H = 255
high_S = 255
high_V = 255
frame_num = 0
frame_diff = 0
frame_first = 0

prev_cen = 0
cen = 110

window_capture_name = 'Frame Capture'
window_detection_name = 'Thresholding Bar'

low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name,window_detection_name,low_H)
    return(low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
    return(high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
    return(low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
    return(high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
    return(low_V)
    
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
    return(high_V)

cap = cv.VideoCapture("lin_vel_13/lin_vel_13.mp4")

#cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

while True:
    low_H = 150
    low_S = 200
    low_V = 110
    # high_H = 236
    # high_S = 255
    # high_V = 255
    ret, frame = cap.read()
    # cv.line(frame,(455,0),(455,1080),(255,0,0),5)
    # cv.line(frame,(805,0),(805,1080),(255,0,0),5)
    # cv.line(frame,(1168,0),(1168,1080),(255,0,0),5)
    cv.imshow("real 1",frame)
    if frame is None:
        break
    frame = frame[500:650, 150:1000]
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    res = cv.bitwise_and(frame, frame, mask=mask)
    locs = np.argwhere(res==255)

    kernel = np.ones((15,15), np.uint8)
    blur = cv.medianBlur(mask,25)
    smoothed = cv.filter2D(blur, -1, kernel)
    dilation = cv.dilate(smoothed,kernel,iterations=1)
    opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)

    res = cv.bitwise_and(frame, frame, mask=opening)
    hsv = cv.cvtColor(res, cv.COLOR_BGR2HSV)
    
    M = cv.moments(dilation)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cent_X.append(cX)
    cent_Y.append(cX)
     
    cv.circle(res, (cX,cY), 10, (255, 255, 255), -1)
    cv.imshow("result",res)
   
    cen = cX
    frame_num  = frame_num + 1
    frame_diff = frame_num - frame_first
    
    if frame_num%4 == 0:
        vel = ((cen - prev_cen) * in_per_pixel)/(frame_diff/fps)
        frame_first = frame_num
        #print(vel)
        vel_set.append(vel)
        frame_set.append(frame_num)
        prev_cen = cen

    key = cv.waitKey(5)
    if key == ord('s'):
        if count == 0:
            f= open("HSV_vals.txt","w+")
            f.write("[Low H, Low S, Low V] - [High H, High S, High V] \r\n")
            f.write("[%d,%d,%d]-" % (low_H, low_S, low_V))
            f.write("[%d,%d,%d]\n" % (high_H, high_S, high_V))
            f.close()
            print("File saved!")
            count = count + 1
        else:
            f= open("HSV_vals.txt","a")
            f.write("[%d,%d,%d]-" % (low_H, low_S, low_V))
            f.write("[%d,%d,%d]\n" % (high_H, high_S, high_V))
            f.close()
            count = count + 1
            print("File saved!")
    elif key == ord('p'):
        print(low_H, low_S,low_V, "\n")
        print(high_H, high_S, high_V,"\n")
    elif key == ord('d'):
        os.remove("HSV_vals.txt")
    elif key == ord('q') or key == 27:
        break

frame_set = np.array(frame_set)
frame_set = (1/60)*frame_set
vel_set = np.array(vel_set)
vel_set = vel_set*0.0254 #converts in/sec to m/sec
vel_std =np.std(vel_set)
vel_avg =np.average(vel_set)

np.savetxt("vel_11.csv", vel_set, delimiter=",")
np.savetxt("frame_11.csv", frame_set, delimiter=",")

plt.plot(frame_set,vel_set, linewidth = 1)
plt.ylabel('Velocity (m/sec)')
plt.ylabel('Time (sec)')
plt.title('Velocity Profile')
plt.show()