def main():
    import numpy as np
    import cv2 as cv
    import argparse
    import os
    import time
    import matplotlib.pyplot as plt

    def make_coordinates(image, line_parameters):
        slope, intercept = line_parameters
        if slope is not None:
            y1 =  image.shape[1]
            y2 = int(y1 * 3/5)
            x1 = int((y1-intercept)/slope)
            x2 = int((y2-intercept)/slope)
        else:
            x1 = 0
            y1 = 0
            x2 = 0
            y2 = 0
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(image, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
            print(left_fit)
            print(left_fit)
        if left_fit is not None:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(image, left_fit_average)
        if right_fit is not None:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line,right_line])

    def canny(image):
        bw = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(bw, (5,5),0)
        canny = cv.Canny(blur, 0, 100)
        return canny

    def region_of_interest(image):
        height = image.shape[0]
        width = image.shape[1]
        polygons = np.array([
        [(50, 360), (640, 360), (640, 200), (150, 150)]
        ])
        mask = np.zeros_like(image)
        cv.fillPoly(mask, polygons, 255)
        masked_image = cv.bitwise_and(image, mask)
        return masked_image

    def display_lines(image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv.line(line_image, (x1, y1), (x2, y2), (255, 0 , 0), 10)
                #(image, line coords, color, thickness)
        return line_image




    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('--camera', help='Camera devide number.', default=0, type=int)
    args = parser.parse_args()

    cap = cv.VideoCapture("best.avi")

    while True:
        _, frame = cap.read()

        if frame is None:
            break
        frame = frame[0:720,0:1280]
        frame = cv.resize(frame,(640,360))
        #frame = frame[:,:,:] + 20

        low_V = 50

        canny_im = canny(frame)

        cropped_image = region_of_interest(canny_im)
        cv.imshow("scrop", cropped_image)    
        lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=50, maxLineGap=5)

        #average_lines = average_slope_intercept(frame,lines)

        line_image = display_lines(frame,lines)
        combined_image = cv.addWeighted(frame,0.8,line_image,1,1)
        cv.imshow("result",combined_image)

        #time.sleep(.025)   # Delays for 0.025 seconds.
        if cv.waitKey(1) & 0XFF== ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

    restart = 1

    if restart == 1:
        main()
    else:
        exit()


main()