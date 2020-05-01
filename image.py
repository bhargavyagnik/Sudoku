
import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while (1):
    _, frame = cap.read()
    #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', frame)
    #cv2.imshow('grey',grey)
    #im=cv2.blur(grey,(5,5))
    #th = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    #kernel = np.ones((2, 2), np.uint8)
    #th2 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    #try:
    #    image, contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    #except:
    #    continue
        
    cv2.imshow('ig',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()


'''
https://github.com/sunnyyeti/Camera-based-real-time-Sudoku-Solver/blob/master/imageProcess.py
http://www.mikedeff.in/sudoku.htm

'''
