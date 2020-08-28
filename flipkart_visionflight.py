import cv2
import imutils
import numpy as np
import pyautogui
import pyzbar.pyzbar as pyzbar #Make sure all these packages are installed

cap = cv2.VideoCapture(0) #Make this 0 if using webcam. 1 if using external camera
u=0 #upward center shift by 5 pixel factor. It is 0 for now.
#cap = cv2.VideoCapture(r'C:\Users\ndale\Downloads\mov.mp4')

U='UP' #These are motion keys of the drone
D='DOWN'
L='LEFT'
R='right'

while(True):
    ret, frame = cap.read() #reading camera feed
    img=frame
    (h, w, d) = img.shape
    reimg=imutils.resize(img,width=700)
    (h, w, d) = reimg.shape #resize the shape of the image obtained to shape of our choice
    center=((w//2)+u,h//2) #upward shifted center of the camera feed

    w1,h1=center[0],center[1]
    #op = cv2.cvtColor(reimg, cv2.COLOR_BGR2GRAY) #converting colour to gray image
    op=cv2.cvtColor(reimg, cv2.COLOR_BGR2HSV) #converting colour to HSV image

    def qr(op):   #This code is for QR Code scanning. Don' touch.
        pic=op.copy()
        info = pyzbar.decode(pic)
        txt=''
        if(len(info)>0):
            for obj in info:
                txt=(obj.data).decode("utf-8")
                (x, y, w, h) = obj.rect
                cv2.rectangle(pic,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.circle(pic,(x+w//2,y+h//2),3,(255,255,0),-1)
            cv2.putText(pic,txt,(40,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        return(pic)

    def makeup(pic):  #This code is for the fixed setup on the camera feed. Don't touch.
        op=pic.copy()
        cv2.rectangle(op,(w1-70,h1+50),(w1+70,h1-50),(0,255,255),2)
        cv2.circle(op,center,7,(255,0,0),2)
        cv2.line(op,(w1+150,h1),(w1+100,h1),(0,255,0),2)
        cv2.line(op,(w1-150,h1),(w1-100,h1),(0,255,0),2)
        cv2.line(op,(w1,h1+70),(w1,h1+100),(0,255,0),2)
        cv2.line(op,(w1,h1-70),(w1,h1-100),(0,255,0),2)
        cv2.putText(op,'L',(w1-200,h1+5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(op,'R',(w1+200,h1+5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(op,'U',(w1-7,h1-120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(op,'D',(w1-7,h1+130),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        return(op)

    def direct(pic,cx,cy): #This code is responsible for adjusting the drone such that its at the rectangle center.
          (h, w, d) = pic.shape
          px,py=w//2,h//2
          if(cx<px-10):
              cv2.putText(pic,'Move LEFT',(px+220,py+5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              #pyautogui.press(L) #Command drone to MOVE RIGHT
          elif(cx>px+10):
              cv2.putText(pic,'Move Right',(px-310,py+5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              #pyautogui.press(R) #Command drone to MOVE LEFT
          elif(cy>py+5):
              cv2.putText(pic,'Move Down',(px-7,py+150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              #pyautogui.press(D) #Command drone to MOVE UP
          elif(cy<py-5):
              cv2.putText(pic,'Move Up',(px-7,py-150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              #pyautogui.press(U) #Command drone to MOVE DOWN
          else:
              cv2.putText(pic,'STAY',(px,py-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              #Command drone to STAY where ever it is.
          return(pic)


    def test(pic,main):   #This code detects the nearest rectangle, its upward shifted center and every other aspect.
        #ret,thresh = cv2.threshold(pic,150,255,cv2.THRESH_BINARY_INV) #Don't touch.
        #image = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
        lower = np.array([22, 93, 0], dtype="uint8")
        upper = np.array([45, 255, 255], dtype="uint8")
        thresh = cv2.inRange(pic, lower, upper)
        contours,_= cv2.findContours(thresh, 1, 2)
        testimg=main.copy()
        area,cen=[],[]
        block=0
        for cnt in contours:
            M = cv2.moments(cnt)
            ar=cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if (len(approx) == 4) and (ar>1000):
                cx,cy = int(M['m10']/M['m00']),int(M['m01']/M['m00'])
                cen.append((cx,cy+u)) #shifted center up
                area.append(ar)
                block=area.index(max(area))

        if(block>0):
            x1,y1=cen[block]
            cv2.circle(testimg,cen[block],3,(0,0,255),-1)
            return(direct(testimg,x1,y1))
        else:
            return(testimg)

    image=makeup(qr(test(op,reimg)))  #This portion initiates the entire code. Don't touch.
    #image=makeup(frame)
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #Don't touch
cv2.destroyAllWindows()
