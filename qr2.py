import serial
#from picamera.array import PiRGBArray
#from picamera import PiCamera
#import trans
import argparse
import time
import cv2
import math
import numpy as np
import zbar
import imutils
def distance(p,q):
    return math.sqrt(math.pow(math.fabs(p[0]-q[0]),2)+math.pow(math.fabs(p[1]-q[1]),2))

def lineEquation(l,m,j):
    a = -((m[1] - l[1])/(m[0] - l[0]))
    b = 1.0
    c = (((m[1] - l[1])/(m[0] - l[0]))*l[0]) - l[1]
    try:
        pdist = (a*j[0]+(b*j[1])+c)/math.sqrt((a*a)+(b*b))
    except:
        #print "LE except"
        return 0
    else:
        return pdist

def lineSlope(l,m):
    dx = m[0] - l[0]
    dy = m[1] - l[1]
    direction = (int(math.atan2(dy,dx) * 180.0 / 3.1415926) + 450) % 360        
    
    if dx != 0: #sw changed from dy as that is the divisor
        align = 1
        dxy = dy/dx
        return dxy,align,direction
    else:
        align = 0
        dxy = 0.0
        return dxy,align,direction

def getSquares(contours,cid):
    x,y,w,h= cv2.boundingRect(contours[cid])
    return x,y,w,h

def updateCorner(p,ref,baseline,corner):
    temp_dist = distance(p,ref)
    if temp_dist > baseline:
        baseline = temp_dist
        corner = p
    return baseline,corner

def getVertices(contours,cid,slope,quad):
    M0 = (0.0,0.0)
    M1 = (0.0,0.0)
    M2 = (0.0,0.0)
    M3 = (0.0,0.0)
    x,y,w,h = cv2.boundingRect(contours[cid])
    A = (x,y)
    B = (x+w,y)
    C = (x+w,h+y)
    D = (x,y+h)
    W = ((A[0]+B[0])/2,A[1])
    X = (B[0],(B[1]+C[1])/2)
    Y = ((C[0]+D[0])/2,C[1])
    Z = (D[0],(D[1]+A[1])/2)
    dmax = []
    for i in range(4):
        dmax.append(0.0)
    pd1 = 0.0
    pd2 = 0.0
    if(slope > 5 or slope < -5 ):
        for i in range(len(contours[cid])):
            pd1 = lineEquation(C,A,contours[cid][i])
            pd2 = lineEquation(B,D,contours[cid][i])
            if(pd1 >= 0.0 and pd2 > 0.0):
                dmax[1],M1 = updateCorner(contours[cid][i],W,dmax[1],M1)
            elif(pd1 > 0.0 and pd2 <= 0):
                dmax[2],M2 = updateCorner(contours[cid][i],X,dmax[2],M2)
            elif(pd1 <= 0.0 and pd2 < 0.0):
                dmax[3],M3 = updateCorner(contours[cid][i],Y,dmax[3],M3)
            elif(pd1 < 0 and pd2 >= 0.0):
                dmax[0],M0 = updateCorner(contours[cid][i],Z,dmax[0],M0)
            else:
                continue
    else:
        halfx = (A[0]+B[0])/2
        halfy = (A[1]+D[1])/2
        for i in range(len(contours[cid])):
            if(contours[cid][i][0][0]<halfx and contours[cid][i][0][1]<=halfy):
                dmax[2],M0 = updateCorner(contours[cid][i][0],C,dmax[2],M0)
            elif(contours[cid][i][0][0]>=halfx and contours[cid][i][0][1]<halfy):
                dmax[3],M1 = updateCorner(contours[cid][i][0],D,dmax[3],M1)
            elif(contours[cid][i][0][0]>halfx and contours[cid][i][0][1]>=halfy):
                dmax[0],M2 = updateCorner(contours[cid][i][0],A,dmax[0],M2)
            elif(contours[cid][i][0][0]<=halfx and contours[cid][i][0][1]>halfy):
                dmax[1],M3 = updateCorner(contours[cid][i][0],B,dmax[1],M3)
    quad.append(M0)
    quad.append(M1)
    quad.append(M2)
    quad.append(M3)
    return quad

def updateCornerOr(orientation,IN):
    if orientation == 0:
        M0 = IN[0]
        M1 = IN[1]
        M2 = IN[2]
        M3 = IN[3]
    elif orientation == 1:
        M0 = IN[1]
        M1 = IN[2]
        M2 = IN[3]
        M3 = IN[0]
    elif orientation == 2:
        M0 = IN[2]
        M1 = IN[3]
        M2 = IN[0]
        M3 = IN[1]
    elif orientation == 3:
        M0 = IN[3]
        M1 = IN[0]
        M2 = IN[1]
        M3 = IN[2]

    OUT = []
    OUT.append(M0)
    OUT.append(M1)
    OUT.append(M2)
    OUT.append(M3)

    return OUT

def cross(v1,v2):
    cr = v1[0]*v2[1] - v1[1]*v2[0]
    return cr

def getIntersection(a1,a2,b1,b2,intersection):
    p = a1
    q = b1
    r = (a2[0]-a1[0],a2[1]-a1[1])
    s = (b2[0]-b1[0],b2[1]-b1[1])
    if cross(r,s) == 0:
        return False, intersection
    t = cross((q[0]-p[0],q[1]-p[1]),s)/float(cross(r,s))
    intersection = (int(p[0]+(t*r[0])),int(p[1]+(t*r[1])))
    return True,intersection
    
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
cv2.namedWindow('rect')    
cap = cv2.VideoCapture(-1)

# Reduce the size of video to 320x240 so rpi can process faster
#cap.set(3,640)
#cap.set(4,480)

# camera = PiCamera()
# camera.resolution = (640,480)
# camera.framerate = 32
# rawCapture = PiRGBArray(camera,size=(640,480))
# time.sleep(0.1)
# for frame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
    # image = frame.array
    # img = image

#show the image
#wait until some key is pressed to procced
oldimg = np.zeros((320,320,3), np.uint8)
while True:
    print "---"
    _, image = cap.read()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)[1]    
    #img = thresh
    edges = cv2.Canny(image,100,200)
    img = image
    #cv2.imshow("thresh",thresh)    
    im2,contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    cv2.imshow("contours",im2)
    mu = []
    mc = []
    mark = 0
    maxc = 0
    bigc = -1
    for x in range(0,len(contours)):
        mu.append(cv2.moments(contours[x]))

    for m in mu:
        if m['m00'] != 0:
            mc.append((m['m10']/m['m00'],m['m01']/m['m00']))
        else:
            mc.append((0,0))
    print hierarchy
    for x in range(0,len(contours)):
        print "h:", x ,hierarchy[0][x] , cv2.contourArea(contours[x])
        k = x
        c = 0
        while(hierarchy[0][k][2] != -1):
            if cv2.contourArea(contours[hierarchy[0][k][2]]) > 5:        
                k = hierarchy[0][k][2]
                c = c + 1
                print "c+1 in loop"
            else:
                k = hierarchy[0][k][2]            
                print "cont too small"
        #if hierarchy[0][k][2] != -1:
        #    c = c + 1
        #    print "c+1 at end"
        if c > maxc:
            maxc = c
            bigc = x
        
        if c > 2:
            print "possible hier1" , x, hierarchy[0][x] , "c:" , c   
            if ((hierarchy[0][x][0] == -1) and  (hierarchy[0][x][1] == -1)):
                print "but rejected as no siblings"
                c = -1
            if (c > 5):
                print "but rejected as too many children descendants"  
                c = -2                
        
        if c > 2:

            ##perimeter = cv2.arcLength(contours[x],True)
            #if (perimeter > 200):
            #    mark = mark #-100
            if mark == 0:
                A = x
            elif mark == 1:
                B = x
            elif mark == 2:
                C = x
            mark = mark+1

    if mark > 2 :
        print "final hier" , A, hierarchy[0][A]    
        print "final hier" , B, hierarchy[0][B]    
        print "final hier" , C, hierarchy[0][C]    
        AB = distance(mc[A],mc[B])
        BC = distance(mc[B],mc[C])
        AC = distance(mc[A],mc[C])

        if(AB>BC and AB>AC):
            outlier = C
            median1 = A
            median2 = B
        elif(AC>AB and AC>BC):
            outlier = B
            median1 = A 
            median2 = C 
        elif(BC>AB and BC>AC):
            outlier = A 
            median1 = B
            median2 = C

        top = outlier
        dist = lineEquation(mc[median1],mc[median2],mc[outlier])
        slope,align,_ = lineSlope(mc[median1],mc[median2])


        if align == 0:
            bottom = median1
            right = median2 
            #orientation = 0 # added by sw as seemed to be missing
        elif(slope < 0 and dist < 0):
            bottom = median1
            right = median2
            orientation = 0
        elif(slope > 0 and dist < 0):
            right = median1
            bottom = median2
            orientation = 1
        elif(slope < 0 and dist > 0):
            right = median1
            bottom = median2
            orientation = 2
        elif(slope > 0 and dist > 0):
            bottom = median1
            right = median2
            orientation = 3
            
        _,_,direction = lineSlope(mc[bottom],mc[top])
        print "direction" , direction    
        areatop = 0.0
        arearight = 0.0
        areabottom = 0.0
        if(top < len(contours) and right < len(contours) and bottom < len(contours) and cv2.contourArea(contours[top]) > 10 and cv2.contourArea(contours[right]) > 10 and cv2.contourArea(contours[bottom]) > 10):
            print "top", top, cv2.contourArea(contours[top]), len(contours)
            tempL = []
            tempM = []
            tempO = []
            src = []
            N = (0,0)
            tempL = getVertices(contours,top,slope,tempL)
            tempM = getVertices(contours,right,slope,tempM)
            tempO = getVertices(contours,bottom,slope,tempO)
            #print tempL
            #print tempM
            #print tempO
            L = updateCornerOr(orientation,tempL)
            M = updateCornerOr(orientation,tempM)
            O = updateCornerOr(orientation,tempO)
            
            iflag,N = getIntersection(M[1],M[2],O[3],O[2],N)
            #print L,M,N,O
            src.append(L[0])
            src.append(M[1])
            src.append(N)
            src.append(O[3])
            src = np.asarray(src,np.float32)
            warped1 = four_point_transform(img,src)
            warped = warped1
            #sw added to rotate image to correct orientation for visual purposes only - not needed for zbar
            #rowsrot,colsrot,dummy = warped1.shape
            #print rowsrot,colsrot,dummy
            #if orientation > 0:
            #    Mrot = cv2.getRotationMatrix2D((colsrot/2,rowsrot/2),(90 * orientation),1)
            #    warped = cv2.warpAffine(warped1,Mrot,(colsrot,rowsrot))
            cv2.imshow("warped",warped)
            cv2.circle(img,N,1,(0,0,255),2)
            cv2.drawContours(img,contours,top,(255,0,0),2)
            cv2.drawContours(img,contours,right,(0,255,0),2)
            cv2.drawContours(img,contours,bottom,(0,0,255),2)
            cv2.drawContours(img,contours,bigc,(255,255,0),2)            
            #print "green:" , cv2.arcLength(contours[bottom],True)
            
            warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
            scanner = zbar.ImageScanner()
            scanner.parse_config('enable')
            imagez = zbar.Image(warped.shape[0],warped.shape[1],'Y800',warped.tostring())
            scanner.scan(imagez)
            for symbol in imagez:
                x = symbol.data
                print x
            #inf =  0 / 0
            oldimg = warped
    else:
        print "no image found"
        cv2.imshow("warped",oldimg)

    cv2.imshow("rect",img)
    #time.sleep(1)
    key = cv2.waitKey(3000) & 0xFF
    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()            