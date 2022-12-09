import face_recognition
import os
import cv2
import numpy as np
from PIL import Image, ImageOps

video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
process_this_frame = True
hat = cv2.imread(os.path.join('img','bonnetdenoel2.png'), cv2.IMREAD_UNCHANGED)
hatPIL = Image.fromarray(hat)
gl_h, gl_w, _ = hat.shape
dbg = False
factor = 2 

# font
font = cv2.FONT_HERSHEY_SIMPLEX
text = "Approchez..." 
org = (50, 50)
fontScale = 1
color = (0, 0, 0) # BGR
thickness = 2 
#logo=cv2.imread("logo_r.png")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1./factor, fy=1./factor)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

    process_this_frame = not process_this_frame
    # Display the results
    imagePIL = Image.fromarray(frame)
    for (top, right, bottom, left) in face_locations :
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= factor
        right *= factor
        bottom *= factor
        left *= factor

        # Draw a box around the face
        if dbg :
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        ratio = (right - left) / gl_w
        #print(gl_w, gl_h)
        (newW, newH) = (int(1.5*gl_w * ratio), int(1.5*gl_h * ratio))

        #newHatPIL = hatPIL.resize((newW, newH))
        newHatPIL = Image.fromarray(cv2.resize(hat, (newW, newH)))
        imagePIL = Image.fromarray(frame)
        imagePIL.paste(newHatPIL,(left, top-newH),newHatPIL)
        frame = np.asarray(imagePIL)

    frame = cv2.flip(frame, 1)
    frame = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    #frame[20:110,550:630:,:]=logo[5:95:,10:90:,:]
    cv2.namedWindow("Joyeux Noël!",cv2.WND_PROP_FULLSCREEN) 
    cv2.setWindowProperty("Joyeux Noël!",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Joyeux Noël!', frame)
    #cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) == ord('q') :
        break
    if cv2.waitKey(1) == ord('d') :
        dbg = not dbg 

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
