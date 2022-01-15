import cv2
import os
from ffpyplayer.player import MediaPlayer


#Face finder
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")


your_file_to_work_with = "assets/Can You Watch This Without Smiling_.mp4" # Add here the path for ur file

asset  = cv2.VideoCapture(your_file_to_work_with) #pass "0" into args to use webcam
# player = MediaPlayer(your_file_to_work_with) #to turn on sound(it douesnt work correctly ,becuse the video becomes slower than the sound (there are different channels)now sure if itll be working with webcam)


#Showing the asset
while True:
    succesfully_read_frame, current_frame = asset.read()

    if not succesfully_read_frame:
        break
        print(">>> Error with detecting current frame!")

    frame_grayscale = cv2.cvtColor(current_frame , cv2.COLOR_BGR2GRAY) #B&W-mode (optimization: RGB has 3 channels ,but B&W only 1)
    
    #Detecting all the faces first
    # * you can delete "scaleFactor=1.5" and "minNeighbors=2" for cleaner resaults, but the FPS will decrease

    faces = face_detector.detectMultiScale(
        frame_grayscale,
        scaleFactor = 1.5,
        minNeighbors = 2
    )

    for (x, y, w, h) in faces: # taking necessery dots
        cv2.rectangle(
            current_frame,
            (x, y),
            (x + w , y + h),
            (100,200,50),
            4
        ) # Drawing rectangle around faces (100,200,50)- color; 4- 4px of sickness of the rectangle
        
        the_face = current_frame[y:y + w , x:x + h]
        face_grayscale = cv2.cvtColor(the_face , cv2.COLOR_BGR2GRAY) # convering to B&W-mode (optimization: RGB has 3 channels, but B&W has only 1)

        #Detecting all the smiles
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 30) #minNeighbors - minimal number of objects, wich the smile contains

        # for (x_smile, y_smile, w_smile, h_smile) in smiles:
        #     cv2.rectangle(the_face,(x_smile,y_smile),(x_smile + w_smile , y_smile + h_smile), (0,0,300), 4) # Drawing rectangle around faces (100,200,50)- color; 4- 4px of sickness of the rectangle

        if len(smiles) > 0:
            cv2.putText(
                current_frame, 
                "smiling", 
                (x, y + h + 40),
                fontScale = 3,
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                color = (255,255,255)
            )


    # window settings
    cv2.imshow("Smile detector", current_frame)


    #Display
    key_pressed= cv2.waitKey(1) #updating frames every 1 ms , if parametes are emty or == 0, then the program will be waiting for key pressing
    space_pressed = False


    #window actions
    if cv2.getWindowProperty('Smile detector', cv2.WND_PROP_VISIBLE) < 1: #windows does close if the user pressed "x" at the right top of the window       
        break
        cv2.destroyAllWindows()

    if key_pressed == 32: #windows does close if the user pressed "Esc"
        space_pressed = True

    #does pause the video
    if space_pressed == True:
        cv2.waitKey(0)
    elif space_pressed == False :
        cv2.waitKey(1)


filename, file_extention = os.path.splitext(your_file_to_work_with) 

if file_extention == ".png" or file_extention == ".jpeg" or file_extention ==".jpg": # If extention is an extention of image, the frame stops and doesnt update.If its video,frames are changing
    cv2.waitKey(0)

# Cleaning up
asset.release()
cv2.destroyAllWindows()