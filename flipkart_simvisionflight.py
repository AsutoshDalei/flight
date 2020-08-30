import cv2
import imutils
import numpy as np
import pyautogui
import autopy
import time
import pydirectinput
from mss import mss
from PIL import Image
import pyzbar.pyzbar as pyzbar
#Make sure all these packages are installed

sct = mss()

time.sleep(5)
u=0 #upward center shift by 5 pixel factor. It is 0 for now.

while(True):
    #ret, frame = cap.read() #Enale these 2 lines to read camera feed
    #reimg=frame

    w,h=410,250 #Dont Change these values. change only if needed.
    monitor = {'top': 130, 'left': 290, 'width': w, 'height': h}
    img = Image.frombytes('RGB', (w,h), sct.grab(monitor).rgb)
    reimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    #op=cv2.cvtColor(np.float32(reimg), cv2.COLOR_BGR2HSV)

    center=((w//2)+u,h//2) #upward shifted center of the camera feed
    w1,h1=center[0],center[1]

    def makeup(pic):  #This code is for the fixed setup on the camera feed. Don't touch.
        op=pic.copy()
        cv2.rectangle(op,(w1-70,h1+50),(w1+70,h1-50),(0,255,255),2)
        cv2.circle(op,center,7,(255,0,0),2)
        cv2.line(op,(w1+150,h1),(w1+100,h1),(0,255,0),2)
        cv2.line(op,(w1-150,h1),(w1-100,h1),(0,255,0),2)
        cv2.line(op,(w1,h1+70),(w1,h1+100),(0,255,0),2)
        cv2.line(op,(w1,h1-70),(w1,h1-100),(0,255,0),2)
        return(op)

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

    def direct(pic,cx,cy): #This code is responsible for adjusting the drone such that its at the rectangle center.
          (h, w, d) = pic.shape
          px,py=w//2,h//2
          if(cx<px-5):
              #cv2.putText(pic,'L',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              pyautogui.press('A')
              #pydirectinput.press('A')
              #autopy.key.tap("A", [autopy.key.Modifier.META])
          elif(cx>px+5):
              #cv2.putText(pic,'R',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              pyautogui.press('D')
              #pydirectinput.press('D')
              #autopy.key.tap("D", [autopy.key.Modifier.META])
          elif(cy>py+2.5):
              #cv2.putText(pic,'D',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              pyautogui.press('S')
              #pydirectinput.press('S')
              #autopy.key.tap("S", [autopy.key.Modifier.META])
          elif(cy<py-2.5):
              #cv2.putText(pic,'U',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              pyautogui.press('W')
              #pydirectinput.press('W')
              #autopy.key.tap("W", [autopy.key.Modifier.META])
          else:
              cv2.putText(pic,'STAY',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
              #Command drone to STAY where ever it is.
          return(pic)


    def test(pic,main):   #This code detects the nearest rectangle, its upward shifted center and every other aspect.
        #ret,thresh = cv2.threshold(pic,150,255,cv2.THRESH_BINARY_INV) #Don't touch.
        image = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)

        lower = np.array([22, 93, 0], dtype="uint8")
        upper = np.array([45, 255, 255], dtype="uint8")

        thresh = cv2.inRange(image, lower, upper)
        contours,_= cv2.findContours(thresh, 1, 2)
        testimg=main.copy()
        area,cen=[],[]
        block=0
        for cnt in contours:
            M = cv2.moments(cnt)
            ar=cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if (len(approx) == 4) and (ar>80):
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

    image=makeup(qr(test(reimg,reimg)))  #This portion initiates the entire code. Don't touch.
    #image=makeup(frame)
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #Don't touch
cv2.destroyAllWindows()
