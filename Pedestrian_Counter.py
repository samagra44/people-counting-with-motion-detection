#import nessasry pakages
import numpy as np
import cv2  
import Person
import time

try:
    log = open('log.txt',"w")
except:
    print( "log file not found")

#Entry and exit counter
cnt_up   = 0
cnt_down = 0

#Source of video
cap = cv2.VideoCapture('Test Files/example_03.mp4')
# cap = cv2.VideoCapture(0)

#for rescaling the frame
def rescale_frame(frame, width, height):
    dim = (width,height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

#Print the capture properties to console
for i in range(19):
    print( i, cap.get(i))

h = 480
w = 640
frameArea = h*w
areaTH = frameArea/250
print( 'Area Threshold', areaTH)

#In and Out lines
line_up = int(2*(h/5))
line_down   = int(3*(h/5))

up_limit =   int(1*(h/5))
down_limit = int(4*(h/5))

print( "Red line y:",str(line_down))
print( "Blue line y:", str(line_up))
line_down_color = (255,0,0)
line_up_color = (0,0,255)
pt1 =  [0, line_down];
pt2 =  [w, line_down];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up];
pt4 =  [w, line_up];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit];
pt6 =  [w, up_limit];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit];
pt8 =  [w, down_limit];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

#Background Substractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#Structuring elements for morphographic filters
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1
while(cap.isOpened()):
   # Read an image from the video source
    ret, frame = cap.read()
    frame = rescale_frame(frame,640,480)

    for i in persons:
        i.age_one() #age every person one frame


    #Apply background subtraction
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    #Binarization to remove shadows (gray color)
    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        #Opening (erode-> dilate) to remove noise
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
        #Closing (dilate -> erode) to join white regions
        mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print( 'UP:',cnt_up)
        print ('DOWN:',cnt_down)
        break

    
    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            # Lack of adding conditions for multi-person, outputs and screen inputs
            
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        # the object is close to one that was detected before
                        new = False
                        i.updateCoords(cx,cy)   #update coordinates on object and resets age
                        if i.going_UP(line_down,line_up) == True:
                            cnt_up += 1;
                            print( "ID:",i.getId(),'crossed going up at',time.strftime("%c"))
                            log.write("ID: "+str(i.getId())+' crossed going up at ' + time.strftime("%c") + '\n')
                        elif i.going_DOWN(line_down,line_up) == True:
                            cnt_down += 1;
                            print( "ID:",i.getId(),'crossed going down at',time.strftime("%c"))
                            log.write("ID: " + str(i.getId()) + ' crossed going down at ' + time.strftime("%c") + '\n')
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        # remove i from the persons list
                        index = persons.index(i)
                        persons.pop(index)
                        del i     #free the memory of i
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
            
    #END for cnt in contours 0
            
    for i in persons:
        cv2.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)
        
    #cnt_store to count person inside the store
    cnt_store = cnt_up - cnt_down
    str_up = 'In: '+ str(cnt_up)
    str_down = 'Out: '+ str(cnt_down)
    str_store = 'In_Store: '+ str(cnt_store)
    frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    cv2.putText(frame, str_up ,(10,400),font,0.5,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,430),font,0.5,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, str_store ,(10,460),font,0.5,(0,255,0),2,cv2.LINE_AA)
    if cnt_store>5:
        cv2.putText(frame,'Warning',(160,260),font,2,(0,0,255),5,cv2.LINE_AA)
    cv2.imshow('Frame',frame)
    cv2.imshow('Mask',mask)    
    

    #press ESC to exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#END while(cap.isOpened())
    
#cleaning
log.flush()
log.close()
cap.release()
cv2.destroyAllWindows()
