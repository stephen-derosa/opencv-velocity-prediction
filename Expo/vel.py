def main():
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
    frame_num = 0
    frame_diff = 0
    frame_first = 0

    prev_cen = 0
    cen = 110

    cap = cv.VideoCapture("lin_vel_13/lin_vel_13.mp4")

    while True:
        low_H = 150
        low_S = 200
        low_V = 110
        high_H = 236
        high_S = 255
        high_V = 255
        rect, frame = cap.read()

        if rect == True:
            cv.imshow("Raw Footage",frame)
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
            cv.imshow("Segmented",res)
        
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
        else:
            break
    restart = 1
    if restart == 1:
        main()
    else:
        exit()
main()