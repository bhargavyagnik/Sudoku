import cv2
import numpy as np
from solve_sudoku import *
from predict import *

def scan_and_solve():
    cap = cv2.VideoCapture(0)
    while True:
        ret,img = cap.read()
        #img=cv2.flip(img,1)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dns=cv2.fastNlMeansDenoising(grey,None,10,10,7)
        dns = cv2.GaussianBlur(dns, (3, 3), 0)
        th3 = cv2.adaptiveThreshold(dns,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)
        th3=255-th3
        contours,_=cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        area =cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        poly=cv2.approxPolyDP(cnt,0.05*perimeter,True)
        if (int(round((np.sqrt(area)/(perimeter/4)),1)*100) in range(90,110)):
            # cv2.drawContours(img,poly, 0, (0, 255, 0), 10)
            # cv2.drawContours(img, poly, 1, (0, 0,255), 10)
            # cv2.drawContours(img, poly, 2, (255, 0, 0), 10)
            # cv2.drawContours(img, poly, 3, (0, 255, 255), 10)
            #print(poly[2][0][0] - poly[0][0][0])

            padding = 5
            e = (padding * 2) + 28
            size=e*9

            if (poly[2][0][0]>= poly[0][0][0]):
                pts1 = np.array([poly[0][0], poly[3][0], poly[1][0], poly[2][0]], np.float32)

                pts2 = np.array([[0, 0], [size, 0], [0, size], [size, size]], np.float32)
            else:
                pts1 = np.array([poly[1][0], poly[0][0], poly[2][0], poly[3][0]], np.float32)
                pts2 = np.array([[0, 0], [size, 0], [0, size], [size, size]], np.float32)
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(th3, M, (size,size))

            grid=np.zeros((9,9))

            for i in range(9):
                for j in range(9):
                    grid[i][j]=no_prediction(dst[(e*i)+padding:(e*(i+1))-padding,padding+(e*(j)):(e*(j+1))-padding])

            cv2.imshow('out', dst)
            print(grid)
            #solve(grid=grid)







            #crop_img = img[y:y+h, x:x+w]

            # print(dst.shape)
            # (x, y, w, h) = cv2.boundingRect(cnt)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #
            # extLeft = cnt[cnt[:, :, 0].argmin()][0]
            # extRight = cnt[cnt[:, :, 0].argmax()][0]
            # extTop = cnt[cnt[:, :, 1].argmin()][0]
            # extBot = cnt[cnt[:, :, 1].argmax()][0]
            # # M = cv2.getPerspectiveTransform((x,y),(x+w,y+h))
            # # dst = cv2.warpPerspective(img, M, (300, 300))
            # # cv2.imshow('out',dst)
            # cv2.circle(img, tuple(extLeft), 3, (0, 0, 255), -1)
            # cv2.circle(img, tuple(extRight), 3, (0, 255, 0), -1)
            # cv2.circle(img, tuple(extTop), 3, (255, 0, 0), -1)
            # cv2.circle(img, tuple(extBot), 3, (255, 255, 0), -1)
            # print("Contour", extLeft,extRight,extTop,extBot)
            # print("Rectangle", (x, y, w, h))


        cv2.imshow('image', img)
        k=cv2.waitKey(30) & 0xff
        if k==27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    scan_and_solve()