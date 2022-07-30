import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

#######################################
wCam, hCam = 640, 480
frameR = 50  # Frame Reduction
smoothening = 7
#######################################

pTime = 0
plocx, plocy = 0, 0
clocx, clocy = 0, 0

cap = cv2.VideoCapture(0)
# changing width and height
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()  # to give width and height of screen

while True:
    # 1. find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)  # to find position of hand

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # give coordinates of index finger
        x2, y2 = lmList[12][1:]  # give coordinates of middle finger
        # print(x1, y1, x2, y2)

        # 3. check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index finger: Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordinates as its moving

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))  # frameR and wCam-frameR will move mouse
            # according to boundary of rectangle drawn
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen values - remove shakiness
            clocx = plocx + (x3-plocx)/smoothening
            clocy = plocy + (y3 - plocy) / smoothening

            # 7. Move mouse
            autopy.mouse.move(wScr - clocx, clocy)  # wScr-clocx done so that mouse will move according to your hand and not in opp.
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocx, plocy = clocx, clocy

        # 8. Both index and middle fingers are up then it's in clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineinfo = detector.findDistance(8, 12, img)  # 8&12 are landmark values
            print(length)

            # 9. Find distance between fingers
            if length < 25:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)

                # 10. Click mouse if distance short
                autopy.mouse.click()

    # 11. Frame rate - speed at which image is shown
    cTime = time.time()
    fps = 1 / (cTime - pTime)  # fps->feet per sec
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display rate

    cv2.imshow("Image", img)
    cv2.waitKey(1)
