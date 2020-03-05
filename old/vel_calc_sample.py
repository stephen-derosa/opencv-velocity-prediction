import numpy as np
import cv2 as cv
import argparse
import os
from time import sleep

count = 0
total_x = 0
total_y = 0

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

#parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
#parser.add_argument('--camera', help='Camera devide number.', default=0, type=int)
#args = parser.parse_args()

#cap = cv.VideoCapture(args.camera)
cap = cv.VideoCapture("lin_vel_2/lin_vel_2.mp4")

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
    cv.imshow("real",frame)
    cv.line(frame,(455,0),(455,1080),(255,0,0),5)
    cv.line(frame,(805,0),(805,1080),(255,0,0),5)
    cv.line(frame,(1168,0),(1168,1080),(255,0,0),5)
    cv.imshow("real",frame)
    if frame is None:
        break
    frame = frame[400:650, 150:1300]
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    res = cv.bitwise_and(frame, frame, mask=mask)
    locs = np.argwhere(res==255)
    #print(mask.tolist())
    
    #print(mask[0]) #(height,width)

    #res= cv.cvtColor(res, cv.COLOR_BGR2HSV)

    kernel = np.ones((5,5), np.uint8)
    #erosion = cv.erode(hsv,kernel,iterations=1)
    blur = cv.medianBlur(mask,25)
    dilation = cv.dilate(blur,kernel,iterations=1)

    #opening - goal false positives and false negatives from image
    opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    #closing = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)
    
    res = cv.bitwise_and(frame, frame, mask=opening)

    #rect,gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    
    M = cv.moments(dilation)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # print(cX)
    # print(cY)
    cv.circle(res, (cX,cY), 10, (255, 255, 255), -1)
    cv.imshow("result",res)

    #speed, calculated every .10 second
    frame_num  = frame_num +1



    #smoothed = cv.filter2D(res, -1, kernel)
    #opening = cv.morphologyEx(smoothed,cv.MORPH_OPEN,kernel)
    # for x in erosion[0]:
    #     for y in erosion[1]:
    #         print(erosion[x,y])
    #         if erosion[x,y]>255:
    #             total_x = x + x
    #             total_y = y + y
    #             count = count + 1 
    #             print(erosion[x,y])

    # for c in cnts:
    #     M = cv.moments(c)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     print(cX)
    #     print(cY)
    #avg_x = total_x/count
    #avg_y = total_y/count

    #cv.circle(erosion, (avg_x,avg_y), 30, (255,255,255), thickness=4, lineType=8, shift=0)

    #cv.imshow("result",res)

    #gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    #cv.imshow("gray", gray)
    #circles = cv.HoughCircles(opening, cv.HOUGH_GRADIENT, 1, 50)
    #median = cv.medianBlur(res,5)

    # circles = np.uint16(np.around(circles))
    # if circles is not None:
    #     for i in circles[0,:]:
	#     #draw the circle in the output image, then draw a rectangle
	#     #corresponding to the center of the circle
	#         cv.circle(opening,(i[0],i[1]),i[2],(255,255,255),3)
    # # print(circles)
    # cv.imshow("result",opening)
    # #cv.imshow(window_capture_name, hsv)
    #cv.imshow(window_detection_name, mask)
    #cv.imshow("frame", frame)
    #cv.imshow("median", median)
    #cv.imshow("smoothed", smoothed)
    #cv.imshow("blurred", blur)


    #cv.imshow("erosion", erosion)
    #cv.imshow("dilation", dilation)
    #cv.imshow("open", opening)
    #cv.imshow("closing", closing)

    #sleep(.5)\

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
        