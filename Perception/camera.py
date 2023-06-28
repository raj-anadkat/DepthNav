import cv2

# Capture frames from /dev/video2
cap = cv2.VideoCapture("/dev/video4")

# Set the resolution to 960x540
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# Set the FPS to 60
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame Cptured !")
        break

    
    cv2.imshow("Frame", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
