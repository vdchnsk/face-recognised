import cv2
import os
from ffpyplayer.player import MediaPlayer
from constants.Settings import (
    KeyboardKeys,
    WindowSettings,
    Images,
)
from constants.Strings import (
    Strings,
)


# Face finder
face_detector = cv2.CascadeClassifier('./core/face_recognition/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('./core/smile_recognition/haarcascade_smile.xml')


your_file_to_work_with = './assets/Can You Watch This Without Smiling_.mp4' # ? Add here the path for your file
# your_file_to_work_with = './assets/Laughing_at_you.jpg' # ? Add here the path for your file

asset  = cv2.VideoCapture(your_file_to_work_with) # ? pass '0' into args to use webcam

# player = MediaPlayer(your_file_to_work_with) # ? to turn on sound(it douesnt work correctly, becuse the video becomes slower than the sound (there are different channels) now sure if itll be working with webcam)


# Showing the asset
while True:
    succesfully_read_frame, current_frame = asset.read()

    if not succesfully_read_frame:
        break
        print('>>> Error with detecting current frame!')

    frame_grayscale = cv2.cvtColor(current_frame , cv2.COLOR_BGR2GRAY) # B&W-mode (optimization: RGB has 3 channels ,but B&W only 1)
    
    # Detecting all the faces first
    # * you can delete 'scaleFactor=1.5' and 'minNeighbors=2' for cleaner resaults, but the FPS will decrease

    faces_detected = face_detector.detectMultiScale(
        frame_grayscale,
        scaleFactor = 1.5,
        minNeighbors = 2
    )

    for (x, y, w, h) in faces_detected: # taking necessery dots
        cv2.rectangle(
            current_frame,
            (x, y),
            (x + w , y + h),
            (100, 200, 50),
            4
        ) # Drawing rectangle around faces (100, 200, 50) - color; 4- 4px of sickness of the rectangle
        
        face_hood = current_frame[y:y + w , x:x + h]
        face_grayscale = cv2.cvtColor(face_hood , cv2.COLOR_BGR2GRAY) # convering to B&W-mode (optimization: RGB has 3 channels, but B&W has only 1)

        smiles_detected = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 30) #minNeighbors - minimal number of objects, wich the smile contains


        if len(smiles_detected) > 0:
            cv2.putText(
                current_frame, 
                Strings.SMILING, 
                (x, y + h + 40),
                fontScale = WindowSettings.WINDOW_FONT_SIZE,
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                color = (255, 255, 255)
            )


    # Window settings
    cv2.imshow(WindowSettings.WINDOW_NAME, current_frame)

    key_pressed= cv2.waitKey(1) # updating frames every 1 ms , if parametes are emty or == 0, then the program will be waiting for key pressing
    space_pressed = False


    if cv2.getWindowProperty(WindowSettings.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: # windows does close if the user pressed 'x' at the right top of the window       
        break
        cv2.destroyAllWindows()

    if key_pressed == KeyboardKeys.SPACE:
        space_pressed = True

    if space_pressed == True:
        cv2.waitKey(0) # Stopping regonising


filename, file_extention = os.path.splitext(your_file_to_work_with) 

if file_extention in Images.SUPPORTED_IMAGE_EXTENSIONS: # If extention is an extention of image, the frame stops and doesnt update.If its video,frames are changing
    cv2.waitKey(0)

# Cleaning up
asset.release()
cv2.destroyAllWindows()