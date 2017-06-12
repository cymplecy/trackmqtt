import cv2
import numpy as np
import paho.mqtt.publish as publish  #import the client1
import time
import math
import ConfigParser

config = ConfigParser.ConfigParser()
config.read("settings.ini")
 
vmin =  int(config.get("HSV", "vmin"))
vmax =  int(config.get("HSV", "vmax"))
smin =  int(config.get("HSV", "smin"))
smax =  int(config.get("HSV", "smax"))
hmin =  int(config.get("HSV", "hmin"))
hmax =  int(config.get("HSV", "hmax"))

ctlx =  int(config.get("correction", "tlx"))
#ctly =  int(config.get("correction", "tly"))
ctrx =  int(config.get("correction", "trx"))
#ctry =  int(config.get("correction", "try"))

cblx =  int(config.get("correction", "blx"))
#cbly =  int(config.get("correction", "bly"))

cbw = int(config.get("correction", "bw"))
cby = int(config.get("correction", "by"))
cty = int(config.get("correction", "ty"))

#cbrx =  int(config.get("correction", "brx"))
#cbry =  int(config.get("correction", "bry"))



def on_connect(client, userdata, flags, rc):
    m="Connected flags"+str(flags)+"result code "\
    +str(rc)+"client1_id  "+str(client)
    print(m)

def on_message(client1, userdata, message):
    print("message received  "  ,str(message.payload.decode("utf-8")))

broker_address="127.0.0.1"
#broker_address="iot.eclipse.org"
# client1 = mqtt.Client("P1")    #create new instance
# client1.on_connect= on_connect        #attach function to callback
# client1.on_message=on_message        #attach function to callback
# time.sleep(1)
# client1.connect(broker_address)      #connect to broker

kernel = np.ones((5,5),np.uint8)

# Take input from webcam
cap = cv2.VideoCapture(-1)

# Reduce the size of video to 320x240 so rpi can process faster
cap.set(3,320)
cap.set(4,240)

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    dst = np.array([
        [0, 0],
        [319, 0],
        [319, 319],
        [0, 319]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (320,320))#maxWidth, maxHeight))

    # return the warped image
    return warped
    


def nothing(x):
    pass
# Creating a windows for later use
cv2.namedWindow('HueComp')
cv2.namedWindow('SatComp')
cv2.namedWindow('ValComp')
cv2.namedWindow('closing')
cv2.namedWindow('Tracking')
cv2.namedWindow('camera')


# Creating track bar for min and max for hue, saturation and value
# You can adjust the defaults as you like
cv2.createTrackbar('hmin', 'HueComp',hmin,179,nothing)
cv2.createTrackbar('hmax', 'HueComp',hmax,179,nothing)

cv2.createTrackbar('smin', 'SatComp',smin,255,nothing)
cv2.createTrackbar('smax', 'SatComp',smax,255,nothing)

cv2.createTrackbar('vmin', 'ValComp',vmin,255,nothing)
cv2.createTrackbar('vmax', 'ValComp',vmax,255,nothing)

#pts = np.array([(ctlx, ctly), (ctrx, ctry), (cblx + cbw, cby), (cblx , cby)])    
cv2.createTrackbar('topleftx', 'Tracking',ctlx,320,nothing)
#cv2.createTrackbar('toplefty', 'Tracking',ctly,240,nothing)
cv2.createTrackbar('toprightx', 'Tracking',ctrx,320,nothing)
#cv2.createTrackbar('toprighty', 'Tracking',ctry,240,nothing)
cv2.createTrackbar('bottomleftx', 'camera',cblx,320,nothing)
cv2.createTrackbar('bottomwidth', 'camera',cbw,320,nothing)
cv2.createTrackbar('bottomy', 'camera',cby,240,nothing)
cv2.createTrackbar('topy', 'Tracking',cty,240,nothing)
#cv2.createTrackbar('bottomlefty', 'camera',cbly,240,nothing)



# My experimental values
# hmn = 12
# hmx = 37
# smn = 145
# smx = 255
# vmn = 186
# vmx = 255

tick = time.time()
dx = 0
dy = 0
wherex = [0,0]
wherey = [0,0]
wherexav = 0
whereyav = 0
oldx = 0
oldy = 0
dia = [0,0]
direction = 90
olddirection = 0
ball=0
msgx = 0
msgy = 0
msgdir = 0
msgdiff = 0
msgradius = 0 
msgbearing = 0
try:
    while(True):
        for ball in range(2):
            _, capframe = cap.read()
            ctlx = cv2.getTrackbarPos('topleftx','Tracking')
            #ctly = cv2.getTrackbarPos('toplefty','Tracking')
            ctrx = cv2.getTrackbarPos('toprightx','Tracking')
            #ctry = cv2.getTrackbarPos('toprighty','Tracking')
            cty = cv2.getTrackbarPos('topy','Tracking')            
            cbrx = cv2.getTrackbarPos('bottomrightx','camera')
            cby = cv2.getTrackbarPos('bottomy','camera')
            cblx = cv2.getTrackbarPos('bottomleftx','camera')
            cbw = cv2.getTrackbarPos('bottomwidth','camera')           
            pts = np.array([(ctlx , cty), (ctrx, cty), (cblx + cbw , cby), (cblx,cby)])    
            frame = four_point_transform(capframe, pts) 
            #frame = cv2.resize(capframe,(320,320))  

            #converting to HSV
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            hue,sat,val = cv2.split(hsv)

            # get info from track bar and appy to result
            hmin = cv2.getTrackbarPos('hmin','HueComp')
            hmax = cv2.getTrackbarPos('hmax','HueComp')
            

            smin = cv2.getTrackbarPos('smin','SatComp')
            smax = cv2.getTrackbarPos('smax','SatComp')


            vmin = cv2.getTrackbarPos('vmin','ValComp')
            vmax = cv2.getTrackbarPos('vmax','ValComp')

            # Apply thresholding
            
            hthresh = cv2.inRange(np.array(hue),np.array(0),np.array(hmin))     
            if ball == 1:
                hthresh = cv2.inRange(np.array(hue),np.array(hmin),np.array(hmax))
           
            sthresh = cv2.inRange(np.array(sat),np.array(smin),np.array(smax))
            vthresh = cv2.inRange(np.array(val),np.array(vmin),np.array(vmax))

            # AND h s and v
            tracking = cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))

            # Some morpholigical filtering
            dilation = cv2.dilate(tracking,kernel,iterations = 1)
            closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
            closing = cv2.GaussianBlur(closing,(5,5),0)
            
            # Detect circles using HoughCircles
            #circles = cv2.HoughCircles(closing,cv2.HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)
            im2,circles,heirach = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # circles = np.uint16(np.around(circles))

            #Draw Circles
            if len(circles) > 0:
                #find largest contour in mask, use to compute minEnCircle 
                c = max(circles, key = cv2.contourArea)
                (x,y), radius = cv2.minEnclosingCircle(c)    
                # If the ball is far, draw it in green
                wherex[ball] = int(x)#(int(round(i[0])) + wherex) / 2
                wherey[ball] = int(y)#(int(round(i[1])) + wherey) / 2
                dia[ball] = int(radius)#(int(round(i[2])) + dia) / 2
                cv2.circle(frame,(wherex[ball],wherey[ball]),dia[ball],(0,255,0),5)
                #client1.loop_start()    #start the loop#
                #client1.subscribe("house/bulbs/bulb1")
            #Show the result in frames
            cv2.imshow('HueComp',hthresh)
            cv2.imshow('SatComp',sthresh)
            cv2.imshow('ValComp',vthresh)
            cv2.imshow('closing',closing)
            cv2.imshow('Tracking',frame)
            cv2.imshow('camera',capframe)    
             
        alphaxy = 0.5 
        wherexav = int((((wherex[0] + wherex[1]) / 2.0) * alphaxy) + (wherexav * (1- alphaxy)))
        whereyav = int((((wherey[0] + wherey[1]) / 2.0) * alphaxy) + (whereyav * (1- alphaxy)))
        radius = int(math.sqrt(((wherexav -160) * (wherexav -160)) + ((120 - whereyav) * (120 - whereyav))))
        alphadir = 0.25
        direction = (int(math.atan2((wherey[0] - wherey[1]),(wherex[0] - wherex[1])) * 180.0 / 3.1415926) + 450) % 360

        diff = (direction - olddirection + 180) % 360 - 180
        diff = diff * alphadir
        direction = int((olddirection + diff + 360) % 360)            
        olddirection = direction
        bearing = ((180 - ((int(math.atan2((120 - whereyav),(wherexav -160)) * 180.0 / 3.1415926) + 450) % 360)) + 360) % 360
        msgs = []
        if (time.time() - tick) > 1:
            for ball in range(2): 
                print "ball"+str(ball)+": ",wherex[ball],wherey[ball]
            msgs = [("where/radius", radius,0,True)]  + msgs              
            msgs = [("where/x", (wherexav -160),0,True)]  + msgs
            msgs = [("where/y",(120 - whereyav),0,True)] + msgs
            print "direction" , direction
            #print "diff", diff
            #msgs = [("where/diff", diff ,0,True)] + msgs
            msgs = [("where/direction", direction,0,True)] + msgs  
            msgs = [("where/bearing", bearing,0,True)] + msgs  

            print msgs
            publish.multiple(msgs, hostname="127.0.0.1")
           
            tick = time.time()
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
except KeyboardInterrupt:
    print ("Keyboard Interrupt")
    
print "exiting prog"

config.set("HSV", "vmin",str(vmin))
config.set("HSV", "vmax",str(vmax))
config.set("HSV", "smin",str(smin))
config.set("HSV", "smax",str(smax))
config.set("HSV", "hmin",str(hmin))
config.set("HSV", "hmax",str(hmax))

config.set("correction", "tlx",str(ctlx))
config.set("correction", "tly",str(ctly))
config.set("correction", "trx",str(ctrx))
config.set("correction", "try",str(ctry))


config.set("correction", "blx",str(cblx))

config.set("correction", "by",str(cby))
config.set("correction", "bw",str(cbw))
config.set("correction", "ty",str(cty)) 
# write changes back to the config file
with open("settings.ini", "wb") as config_file:
    config.write(config_file)  



cap.release()

cv2.destroyAllWindows()