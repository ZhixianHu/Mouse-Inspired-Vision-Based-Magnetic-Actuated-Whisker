import numpy as np
import cv2
import pandas as pd

polygonMask = cv2.imread("img/polygonMask.png", cv2.IMREAD_GRAYSCALE)
polygonMask = cv2.merge([polygonMask, polygonMask, polygonMask])

# This function is used to reorder the points in the list
def listReOrder(ptList):
    if ptList[0][0] < ptList[1][0]:
        mid = ptList[0]
        ptList[0] = ptList[1]
        ptList[1] = mid
    if ptList[2][0] < ptList[3][0]:
        mid = ptList[2]
        ptList[2] = ptList[3]
        ptList[3] = mid
    if ptList[4][0] < ptList[5][0]:
        mid = ptList[4]
        ptList[4] = ptList[5]
        ptList[5] = mid
    if ptList[6][0] < ptList[7][0]:
        mid = ptList[6]
        ptList[6] = ptList[7]
        ptList[7] = mid

# This function is used to reorder the points in the list
def rearrangeList(ptList):
    newList = [None] * 8
    newList[0] = ptList[0]
    newList[1] = ptList[1]
    newList[2] = ptList[3]
    newList[3] = ptList[5]
    newList[4] = ptList[7]
    newList[5] = ptList[6]
    newList[6] = ptList[4]
    newList[7] = ptList[2]
    return newList

# This function is used to undistort the image
def imgUndistort(frame, calibParaPath = "img/calibration_data.npz"):
    calibPara = np.load(calibParaPath)
    mtx = calibPara['mtx']
    dist = calibPara['dist']
    # rvecs = calibPara['rvecs']
    # tvecs = calibPara['tvecs']
    height, width = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    frame = frame[y:y+h, x:x+w]
    return frame

def imgPreMaskProcess(frame, mask):
    frame = cv2.bitwise_and(frame, mask)
    return frame

def imgPreProcess(frame):
    frame = imgPreMaskProcess(frame, polygonMask)
    frame = imgUndistort(frame)
    return frame

def imgRedFindCentroid(frame, minArea = 10, mskID = 0):
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if mskID == 0:
        mskRedThr1_low = np.array([0, 80, 60])
        mskRedThr1_high = np.array([10, 255, 255])
        mskRedThr2_low = np.array([170, 80, 60])
        mskRedThr2_high = np.array([180, 255, 255])
        
        msk1 = cv2.inRange(hsvImg, mskRedThr1_low, mskRedThr1_high)
        msk2 = cv2.inRange(hsvImg, mskRedThr2_low, mskRedThr2_high)
        mask = msk1 + msk2
        
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask, kernel, iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 1)
        mask = dilation
    elif mskID == 1:
        mskRedThr1_low = np.array([0, 80, 60])
        mskRedThr1_high = np.array([10, 255, 255])
        mskRedThr2_low = np.array([170, 80, 60])
        mskRedThr2_high = np.array([180, 255, 255])
        
        msk1 = cv2.inRange(hsvImg, mskRedThr1_low, mskRedThr1_high)
        msk2 = cv2.inRange(hsvImg, mskRedThr2_low, mskRedThr2_high)
        mask = msk1 + msk2
        
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask, kernel, iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 2)
        erosion = cv2.erode(dilation, kernel, iterations = 1)   
        mask = erosion

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filteredContours = []
    centerPtList = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= minArea:
            filteredContours.append(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centerPtList.append((cX, cY))

    if len(filteredContours) != 8:
        # print("-----!!! Contour detection failed.--------")
        # print("contour: ", len(contours), "mskID: ", mskID)
        centerPtList = []
        # cv2.imshow("Img Red", mask)
        # cv2.imshow('Original Frame', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if mskID == 1:
            print("-----!!! Contour detection failed.--------")
            print("contour: ", len(contours), "mskID: ", mskID)
            return mask, centerPtList, False
        return imgRedFindCentroid(frame, minArea, mskID+1)
    # else:
    #     cv2.imshow("Img Red", mask)
    #     cv2.imshow('Original Frame', frame)
    #     cv2.waitKey(2000)
    # cv2.destroyAllWindows()

    listReOrder(centerPtList)

    # Visualize the result
    # trkImg = frame.copy()
    # redMask = np.zeros_like(mask, dtype=np.uint8)
    # cv2.drawContours(redMask, filteredContours, -1, 255, thickness=cv2.FILLED)
    # result = cv2.bitwise_and(frame, frame, mask=redMask)

    # for i in range(8):
    #     cv2.circle(trkImg, orgCenterPtList[i], 3, (255, 255, 255), -1)
    #     cv2.circle(redMask, orgCenterPtList[i], 3, (0, 255, 0), -1)
    # cv2.imshow("Track", trkImg)
    # cv2.imshow("Img Red", mask)
    # cv2.imshow('Original Frame', frame)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    return mask, centerPtList, True

orgImg = cv2.imread("img/org.png")
orgImg = imgPreProcess(orgImg)

orgRedMask, orgCenterPtList, orgRedFindFlag = imgRedFindCentroid(orgImg)
orgCenterPtList = rearrangeList(orgCenterPtList)

curveFitCenterPtList = [
    (282, 297),  
    (190, 287), 
    (345, 233),
    (130, 220), 
    (353,  142), 
    (135, 127), 
    (295, 74),
    (202, 66), 
]
curveFitCenterPtList = rearrangeList(curveFitCenterPtList)

curve_fit_para = [
    [  0.79557368, 282.3508395, 297.03295553,  -0.42777786],
    [  0.89075305, 190.45936981, 286.96051867,  -0.35218606],
    [  0.91560196, 345.01432358, 232.57820699,  -3.61919832],
    [  0.85412267, 130.00728004, 219.92369247,  -0.24930196],
    [  0.870223958, 353.435857,  142.422444, -0.334622223],
    [  0.87530027, 134.67438286, 127.0337014,   -0.17724465],
    [  0.830036046, 295.187737, 73.6946555, -0.218070369],
    [  0.821017368, 202.121623, 66.0259798, 0.068201556]
]
curve_fit_para = rearrangeList(curve_fit_para)

def imgRedCentroidTrack(frame, orgCenterPtL, imgPath, minArea = 10):
    org = frame.copy()
    trkImg = frame.copy()
    redMask, centerPtList, redFindFlag = imgRedFindCentroid(frame, minArea)
    centerPtList = rearrangeList(centerPtList)
    int_orgCenterPtL = [(int(pt[0]), int(pt[1])) for pt in orgCenterPtL]
    if not redFindFlag:
        print(imgPath)
        centerPtList = []
        cv2.imshow("RedCentroid Track", trkImg)
        cv2.imshow("RedCentroid Img Red", redMask)
        cv2.imshow('RedCentroid Original Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return frame, redMask, org, centerPtList, [], []
    
    mask = np.zeros_like(redMask, dtype=np.uint8)
    dx = []
    dy = []
    for i in range(8):
        cv2.circle(trkImg, int_orgCenterPtL[i], 3, (255, 255, 255), -1)
        cv2.circle(trkImg, centerPtList[i], 3, (255, 255, 0), -1)
        cv2.circle(mask, centerPtList[i], 3, (0, 255, 0), -1)
        cv2.arrowedLine(trkImg, pt1=int_orgCenterPtL[i], pt2 = centerPtList[i], color=(0,255,255),\
                        thickness = 2, line_type = cv2.LINE_8, shift=0, tipLength=0.5)
        dx.append(centerPtList[i][0] - orgCenterPtL[i][0])
        dy.append(centerPtList[i][1] - orgCenterPtL[i][1])
    dx = np.array(dx)
    dy = np.array(dy)
    return frame, redMask, org, centerPtList, dx, dy

