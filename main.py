import cv2

vid_window = cv2.VideoCapture(0)

while True:
    ret, image=vid_window.read()
    cv2.imshow("Hand Pose", image)

    key = cv2.waitkey(1)
    if k==ord('n'):
        break
    elif cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
    
video.release()
cv2.destroyAllWindows()
