import cv2

#Face finder
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Webcome grabbing
asset  = cv2.VideoCapture("assets/Can You Watch This Without Smiling_.mp4")

#Showing the asset
while True:
    #reads the current frame of the asset or the webcam stream
    succesfully_read_frame , frame = asset.read()
    if not succesfully_read_frame:
        break
    # window settingd
    cv2.imshow("Smile detector",frame)

    #Display
    cv2.waitKey(40) #updating frames every 40 ms , if parametes are emty or == 0, then the program will be waiting for key pressing


#Clean up
asset.release() #OS gives us a permisiion to use camera/video-asset e.t.c (I got it so)
cv2.destroyAllWindows() #destroyong all the recently opened smile detector windows