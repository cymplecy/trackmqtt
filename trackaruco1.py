import cv2
import numpy as np
import paho.mqtt.publish as publish  #import the client1
import time
import math
import ConfigParser
import aruco

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
    #print M
    #time.sleep(30)
    warped = cv2.warpPerspective(image, M, (320,320))#maxWidth, maxHeight))

    # return the warped image
    return warped
    


def nothing(x):
    pass
# Creating a windows for later use
#cv2.namedWindow('HueComp')
#cv2.namedWindow('SatComp')
#cv2.namedWindow('ValComp')
#cv2.namedWindow('closing')
cv2.namedWindow('Tracking')
cv2.namedWindow('camera')


# Creating track bar for min and max for hue, saturation and value
# You can adjust the defaults as you like
#cv2.createTrackbar('hmin', 'HueComp',hmin,179,nothing)
#cv2.createTrackbar('hmax', 'HueComp',hmax,179,nothing)

#cv2.createTrackbar('smin', 'SatComp',smin,255,nothing)
#cv2.createTrackbar('smax', 'SatComp',smax,255,nothing)

#cv2.createTrackbar('vmin', 'ValComp',vmin,255,nothing)
#cv2.createTrackbar('vmax', 'ValComp',vmax,255,nothing)

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

# load board and camera parameters
#boardconfig = aruco.BoardConfiguration("chessboardinfo_small_meters.yml")
camparam = aruco.CameraParameters()
camparam.readFromXMLFile("dfk72_6mm_param2.yml")

# create detector and set parameters
detector = aruco.MarkerDetector()
params = detector.getParams()

#detector.setParams(camparam)
# set minimum marker size for detection
#markerdetector = detector.getMarkerDetector()
#markerdetector.setMinMaxSize(0.01)
    
try:
    _,camframe = cap.read()
    # Apply thresholding
    ctlx = cv2.getTrackbarPos('topleftx','Tracking')
    #ctly = cv2.getTrackbarPos('toplefty','Tracking')
    ctrx = cv2.getTrackbarPos('toprightx','Tracking')
    #ctry = cv2.getTrackbarPos('toprighty','Tracking')
    cty = cv2.getTrackbarPos('topy','Tracking')            
    cbrx = cv2.getTrackbarPos('bottomrightx','camera')
    cby = cv2.getTrackbarPos('bottomy','camera')
    cblx = cv2.getTrackbarPos('bottomleftx','camera')
    cbw = cv2.getTrackbarPos('bottomwidth','camera')           

    ctly = cty
    ctry = cty
    crby = cby
    clby = cby
    
    #frame = four_point_transform(capframe, pts) 
    #frame = cv2.resize(capframe,(320,320))          
    markers = detector.detect(camframe)

    for marker in markers:
        if marker.id < 10:        
            # print marker ID and point positions
            print("Marker: {:d}".format(marker.id))
            print "green", marker[1], type(marker[1])
            if marker.id == 1:
                ctlx = int(marker[2].item(0)) 
                ctly = int(marker[2].item(1))
            if marker.id == 2:
                ctrx = int(marker[3].item(0))
                ctry = int(marker[3].item(1))              
            if marker.id == 3:
                cbrx = int(marker[0].item(0)) 
                cbry = int(marker[0].item(1))
            if marker.id == 4:
                cblx = int(marker[1].item(0))
                cbly = int(marker[1].item(1))                
            for i, point in enumerate(marker):
                print("\t{:d} {}".format(i, str(point)))
            marker.draw(camframe, np.array([255, 255, 255]), 2)

            # # calculate marker extrinsics for marker size of 3.5cm
            # marker.calculateExtrinsics(0.035, camparam)
            # print("Marker extrinsics:\n{:s}\n{:s}".format(marker.Tvec, marker.Rvec))

            print("detected ids: {}".format(", ".join(str(m.id) for m in markers)))         

    cby = cbly
    cbw = cbrx - cblx
    cty = (ctly + ctry) / 2
    
    pts = np.array([(ctlx , cty), (ctrx, cty), (cblx + cbw , cby), (cblx,cby)]) 
    tick = time.time()
    wherex = 0
    wherey = 0
    directon = 0
    while(True):

        _, capframe = cap.read()

        frame = four_point_transform(capframe, pts) 
        #frame = cv2.resize(capframe,(320,320))  


        # Apply thresholding
        
        markers = detector.detect(frame)

        for marker in markers:
            # print marker ID and point positions
            if marker.id > 9:
                #print("Marker: {:d}".format(marker.id))
                wherex = 0
                for loop in range(4):
                    wherex += marker[loop].item(0)
                wherex = int(wherex / 4)
                wherey = 0
                for loop in range(4):
                    wherey += marker[loop].item(1)
                wherey = int(wherey / 4)                
                print wherex,wherey
                direction = (int(math.atan2((marker[1].item(1) - marker[2].item(1) ),(marker[1].item(0) - marker[2].item(0) )) * 180.0 / 3.1415926) + 450) % 360
                marker.draw(frame, np.array([255, 255, 255]), 2)


        #print("detected ids: {}".format(", ".join(str(m.id) for m in markers)))
             
            # alphaxy = 0.5 
            # wherexav = int((((wherex[0] + wherex[1]) / 2.0) * alphaxy) + (wherexav * (1- alphaxy)))
            # whereyav = int((((wherey[0] + wherey[1]) / 2.0) * alphaxy) + (whereyav * (1- alphaxy)))
            # radius = int(math.sqrt(((wherexav -160) * (wherexav -160)) + ((120 - whereyav) * (120 - whereyav))))
            # alphadir = 0.25
            # direction = (int(math.atan2((wherey[0] - wherey[1]),(wherex[0] - wherex[1])) * 180.0 / 3.1415926) + 450) % 360

            # diff = (direction - olddirection + 180) % 360 - 180
            # diff = diff * alphadir
            # direction = int((olddirection + diff + 360) % 360)            
            # olddirection = direction
            # bearing = ((180 - ((int(math.atan2((120 - whereyav),(wherexav -160)) * 180.0 / 3.1415926) + 450) % 360)) + 360) % 360
        msgs = []
        if (time.time() - tick) > 1:
            #msgs = [("where/radius", radius,0,True)]  + msgs              
            msgs = [("where/x", (wherex -160),0,True)]  + msgs
            msgs = [("where/y",(160 - wherey),0,True)] + msgs
            #print "direction" , direction
            #print "diff", diff
            #msgs = [("where/diff", diff ,0,True)] + msgs
            msgs = [("where/direction", direction,0,True)] + msgs  
            #msgs = [("where/bearing", bearing,0,True)] + msgs  

            print msgs
            publish.multiple(msgs, hostname="127.0.0.1")
               
            tick = time.time()
        #cv2.imshow("frame", frame)    
        cv2.imshow("Tracking", frame)
        cv2.imshow("camera", camframe)   
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