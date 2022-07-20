import cv2 as cv
import hand_tracking as ht
from api_request import FireRise 
from api_request import FireRise

def videoCapture():
    # Camera capture
    cap         = cv.VideoCapture(2)
    i           = 0
    tracking    = ht.handDetector(detectionCon=0.75, maxHands=1)
    ids         = [4, 8, 12, 16, 20]

    if(cap.isOpened() == False):
        print("Error openning the video")

    while(cap.isOpened()):
        success, frame  = cap.read()
        contour         = tracking.findHands(frame)
        pose            = tracking.findPosition(frame)
        i               += 1

        if len(pose) != 0:
            fingers = []
            
            #print(tracking.label)

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
            api = FireRise("https://myhand-ff333-default-rtdb.firebaseio.com/", fingers)
            api.putData("mao", True, None, fingers)

        if success:
            cv.imshow('Frame', cv.flip(frame, 1))
            key = cv.waitKey(1)

            if key == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    videoCapture()