import cv2
import os

#Face finder
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")
#Webcome/video  grabbing

your_file_to_work_with = "assets/Laughing_at_you.jpg" #Add here the path for ur file
asset  = cv2.VideoCapture(your_file_to_work_with) #pass "0" into args to use webcam
#Showing the asset
while True:
    succesfully_read_frame , frame = asset.read() #reads the current frame of the asset or the webcam stream
    if not succesfully_read_frame: #Looks for the errors and stops the apps work if they are
        break
        print("Error with detecting some frame")

    frame_grayscale = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) #B&W-mode (optimization: RGB has 3 channels ,but B&W only 1)
    
    #Detect all the faces first
    faces = face_detector.detectMultiScale(frame_grayscale, scaleFactor=1.5, minNeighbors=2) #delete "scaleFactor=1.5, minNeighbors=2" for better resault,but the video'll become much more slower

    for (x, y, w, h) in faces: # taking dots from the lidt "faces"
        cv2.rectangle(frame,(x,y),(x+w , y+h), (100,200,50), 4) # Drawing rectangle around faces (100,200,50)- color; 4- 4px of sickness of the rectangle
        
        the_face = frame[y:y + w , x:x + h] # slicing the list (NumPy feature)
        face_grayscale = cv2.cvtColor(the_face , cv2.COLOR_BGR2GRAY) #B&W-mode (optimization: RGB has 3 channels ,but B&W only 1)

        #Detect all the smiles
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=30) #minNeighbou - minimal number of objects, wich the smile contains

        # for (x_smile, y_smile, w_smile, h_smile) in smiles:
        #     cv2.rectangle(the_face,(x_smile,y_smile),(x_smile + w_smile , y_smile + h_smile), (0,0,300), 4) # Drawing rectangle around faces (100,200,50)- color; 4- 4px of sickness of the rectangle

        if len(smiles)>0:
            cv2.putText(frame, "person", (x, y+h+40), fontScale = 3, fontFace = cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
    # window settings
    cv2.imshow("Smile detector",frame)

    #Display
    key_pressed= cv2.waitKey(1) #updating frames every 1 ms , if parametes are emty or == 0, then the program will be waiting for key pressing
    space_pressed = False
     
    #window actions
    if cv2.getWindowProperty('Smile detector',cv2.WND_PROP_VISIBLE) < 1: #windows does close if the user pressed "x" at the right top of the window       
        break
        cv2.destroyAllWindows()

    if key_pressed == 32:#windows does close if the user pressed "Esc"
        space_pressed = True

        #does pause the video
    if space_pressed == True:
        cv2.waitKey(0)
    elif space_pressed == False :
        cv2.waitKey(1)

    #detectong extention of passed file to work with.Actions with it
filename, file_extention = os.path.splitext(your_file_to_work_with) #spliting the fileName and its extention
if file_extention == ".png" or file_extention == ".jpeg" or file_extention ==".jpg":# If extention is an extention of image, the frame stops and doesnt update.If its video,frames are changing
    cv2.waitKey(0)


#Cleaning up
asset.release() #OS gives us a permisiion to use camera/video-asset e.t.c (I got it so)
cv2.destroyAllWindows() #destroyong all the recently opened smile detector windows