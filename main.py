import cv2

#Face finder
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")
#Webcome grabbing
asset  = cv2.VideoCapture("assets/Can You Watch This Without Smiling_.mp4")

#Showing the asset
while True:
    succesfully_read_frame , frame = asset.read() #reads the current frame of the asset or the webcam stream
    if not succesfully_read_frame: #Looks for the errors and stops the apps work if they are
        break

    frame_grayscale = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) #B&W-mode (optimization: RGB has 3 channels ,but B&W only 1)
    
    #Detect all the faces first
    faces = face_detector.detectMultiScale(frame_grayscale)
    #Detect all the smiles
    smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor=1.7, minNeighbors=30)#minNeighbou - minimal number of objects, wich the smile contains
    
    for (x, y, w, h) in faces: # taking dots from the lidt "faces"
        cv2.rectangle(frame,(x,y),(x+w , y+h), (100,200,50), 4) # Drawing rectangle around faces (100,200,50)- color; 4- 4px of sickness of the rectangle

    for (x, y, w, h) in smiles: # taking dots from the lidt "faces"
        cv2.rectangle(frame,(x,y),(x+w , y+h), (0,0,300), 4) # Drawing rectangle around faces (100,200,50)- color; 4- 4px of sickness of the rectangle
        
    # window settings
    cv2.imshow("Smile detector",frame)

    #Display
    cv2.waitKey(1) #updating frames every 1 ms , if parametes are emty or == 0, then the program will be waiting for key pressing


#Clean up
asset.release() #OS gives us a permisiion to use camera/video-asset e.t.c (I got it so)
cv2.destroyAllWindows() #destroyong all the recently opened smile detector windows