from distutils.command.sdist import sdist
import cv2 as cv
import time
import os
import hand_tracking as ht

# Camera capture
cap         = cv.VideoCapture(1)
i           = 0
tracking    = ht.handDetector(detectionCon=0.75)
ids         = [4, 8, 12, 16, 20]

if(cap.isOpened() == False):
    print("Error openning the video")
else: 
    # Frame rate info
    fps = cap.get(5)
    print('Frames per second: ', fps, 'FPS')

    # Frame count
    frame_count = cap.get(7)
    print('Frame Count: ', frame_count)


while(cap.isOpened()):
    ret, frame  = cap.read()
    contour     = tracking.findHands(frame)
    pose        = tracking.findPosition(frame)
    i          += 1

    if len(pose) != 0:
        fingers = []
        
        print(tracking.label)

        if tracking.label == 'Left':
            # hand Thumb -> Left
            if pose[ids[0]][1] > pose[ids[0] - 1][1]:
                    fingers.append(1)
            else: 
                fingers.append(0)

        elif tracking.label == 'Right':
            # hand Thumb -> Right
            if pose[ids[0]][1] < pose[ids[0] - 1][1]:
                    fingers.append(1)
            else: 
                fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            # Check finger reference points to define hand is open or not
            if pose[ids[id]][2] < pose[ids[id] - 2][2]:
                fingers.append(1)
            else: 
                fingers.append(0)
        
        print(fingers)


    if ret:
        cv.imshow('Frame', frame)
        key = cv.waitKey(20)

        if key == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()